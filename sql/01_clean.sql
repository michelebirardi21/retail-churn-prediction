-- 01_clean.sql
-- Input : transactions_raw (creata da src/load_data.py, fogli già uniti e senza sovrapposizioni)
-- Output: transactions_clean (acquisti + resi puliti), purchases, returns, cleaning_report

-- ---------------------------------------------------------------
-- 1. Tabella pulita
-- ---------------------------------------------------------------
CREATE OR REPLACE TABLE transactions_clean AS
SELECT DISTINCT
    Invoice                                   AS invoice,
    UPPER(TRIM(StockCode))                    AS stock_code,
    TRIM(Description)                         AS description,
    Quantity                                  AS quantity,
    InvoiceDate                               AS invoice_ts,
    CAST(InvoiceDate AS DATE)                 AS invoice_date,
    Price                                     AS unit_price,
    CAST("Customer ID" AS BIGINT)             AS customer_id,   -- da DOUBLE (13085.0) a intero
    Country                                   AS country,
    Invoice LIKE 'C%'                         AS is_return,
    Quantity * Price                          AS line_amount
FROM transactions_raw
WHERE "Customer ID" IS NOT NULL                              -- senza cliente non si può costruire lo storico
  AND regexp_matches(UPPER(TRIM(StockCode)), '^[0-9]{5}')    -- solo prodotti veri (esclude POST, DOT, M, BANK CHARGES, ...)
  AND Price > 0
  AND (
        (Invoice NOT LIKE 'C%' AND Quantity > 0)             -- acquisti
     OR (Invoice LIKE 'C%'     AND Quantity < 0)             -- resi / annullamenti
  );

-- ---------------------------------------------------------------
-- 2. Viste comode
-- ---------------------------------------------------------------
CREATE OR REPLACE VIEW purchases AS
SELECT * FROM transactions_clean WHERE NOT is_return;

CREATE OR REPLACE VIEW returns AS
SELECT * FROM transactions_clean WHERE is_return;

-- ---------------------------------------------------------------
-- 3. Report di pulizia (da copiare nel README)
--    I conteggi sono per regola, non esclusivi: una riga può violarne più d'una.
-- ---------------------------------------------------------------
CREATE OR REPLACE TABLE cleaning_report AS
SELECT 1 AS ord, 'righe totali (dopo dedup fogli)' AS step, COUNT(*) AS n_rows
FROM transactions_raw
UNION ALL
SELECT 2, 'senza Customer ID', COUNT(*)
FROM transactions_raw WHERE "Customer ID" IS NULL
UNION ALL
SELECT 3, 'StockCode non prodotto (POST, DOT, M, ...)', COUNT(*)
FROM transactions_raw WHERE NOT regexp_matches(UPPER(TRIM(StockCode)), '^[0-9]{5}')
UNION ALL
SELECT 4, 'prezzo <= 0', COUNT(*)
FROM transactions_raw WHERE Price <= 0
UNION ALL
SELECT 5, 'quantità incoerente col tipo di fattura', COUNT(*)
FROM transactions_raw
WHERE (Invoice NOT LIKE 'C%' AND Quantity <= 0) OR (Invoice LIKE 'C%' AND Quantity >= 0)
UNION ALL
SELECT 6, 'duplicati esatti rimossi (stima su righe valide)',
       (SELECT COUNT(*) FROM transactions_raw
        WHERE "Customer ID" IS NOT NULL
          AND regexp_matches(UPPER(TRIM(StockCode)), '^[0-9]{5}')
          AND Price > 0
          AND ((Invoice NOT LIKE 'C%' AND Quantity > 0) OR (Invoice LIKE 'C%' AND Quantity < 0)))
       - (SELECT COUNT(*) FROM transactions_clean)
UNION ALL       
SELECT 7, 'righe finali in transactions_clean', COUNT(*)
FROM transactions_clean
ORDER BY ord;
