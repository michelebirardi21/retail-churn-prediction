-- 02_features.sql
-- Una riga per cliente, calcolata "come se oggi fosse $cutoff".
--   * feature: usano SOLO transazioni con invoice_date < $cutoff  (niente leakage)
--   * etichetta churn: 1 se il cliente NON ha spesa netta positiva nei 90 giorni a partire da $cutoff
-- Parametro: $cutoff (data). Viene passato da src/build_dataset.py.
-- Popolazione: clienti attivi (almeno un acquisto negli ultimi 180 giorni prima del taglio).

WITH params AS (
    SELECT CAST($cutoff AS DATE) AS cutoff
),
p AS (  -- acquisti prima del taglio
    SELECT t.* FROM purchases t, params WHERE t.invoice_date < params.cutoff
),
r AS (  -- resi prima del taglio
    SELECT t.* FROM returns t, params WHERE t.invoice_date < params.cutoff
),
days AS (  -- giorni distinti in cui il cliente ha comprato
    SELECT customer_id, invoice_date
    FROM p
    GROUP BY customer_id, invoice_date
),
gaps AS (
    SELECT customer_id,
           invoice_date,
           date_diff('day',
                     LAG(invoice_date) OVER (PARTITION BY customer_id ORDER BY invoice_date),
                     invoice_date) AS gap_days
    FROM days
),
gap_stats AS (
    SELECT customer_id,
           AVG(gap_days)                  AS avg_gap_days,
           STDDEV_SAMP(gap_days)          AS std_gap_days,
           arg_max(gap_days, invoice_date) AS last_gap_days
    FROM gaps
    WHERE gap_days IS NOT NULL
    GROUP BY customer_id
),
agg AS (
    SELECT
        p.customer_id,
        MIN(p.invoice_date)               AS first_date,
        MAX(p.invoice_date)               AS last_date,
        COUNT(DISTINCT p.invoice)         AS n_orders,
        COUNT(DISTINCT p.invoice_date)    AS n_purchase_days,
        COUNT(DISTINCT p.stock_code)      AS n_products,
        SUM(p.line_amount)                AS total_spend,
        -- finestre recenti
        COUNT(DISTINCT CASE WHEN p.invoice_date >= params.cutoff - 30 THEN p.invoice END) AS orders_30d,
        COUNT(DISTINCT CASE WHEN p.invoice_date >= params.cutoff - 60 THEN p.invoice END) AS orders_60d,
        COUNT(DISTINCT CASE WHEN p.invoice_date >= params.cutoff - 90 THEN p.invoice END) AS orders_90d,
        COALESCE(SUM(CASE WHEN p.invoice_date >= params.cutoff - 30 THEN p.line_amount END), 0) AS spend_30d,
        COALESCE(SUM(CASE WHEN p.invoice_date >= params.cutoff - 60 THEN p.line_amount END), 0) AS spend_60d,
        COALESCE(SUM(CASE WHEN p.invoice_date >= params.cutoff - 90 THEN p.line_amount END), 0) AS spend_90d,
        -- i 90 giorni precedenti a quelli recenti (da -180 a -90), per misurare il trend
        COUNT(DISTINCT CASE WHEN p.invoice_date >= params.cutoff - 180
                             AND p.invoice_date <  params.cutoff - 90 THEN p.invoice END) AS orders_prev_90d,
        COALESCE(SUM(CASE WHEN p.invoice_date >= params.cutoff - 180
                           AND p.invoice_date <  params.cutoff - 90 THEN p.line_amount END), 0) AS spend_prev_90d,
        -- B2B vs B2C
        AVG(p.quantity)                   AS avg_qty_per_line,
        AVG(p.unit_price)                 AS avg_unit_price,
        -- finestra fissa 180 giorni (stessa scala tra train 1 e test)
        COUNT(DISTINCT CASE WHEN p.invoice_date >= params.cutoff - 180 THEN p.invoice END) AS orders_180d,
        COALESCE(SUM(CASE WHEN p.invoice_date >= params.cutoff - 180 THEN p.line_amount END), 0) AS spend_180d,
    FROM p CROSS JOIN params
    GROUP BY p.customer_id
),
ret AS (
    SELECT customer_id,
           COUNT(DISTINCT invoice) AS n_return_invoices,
           -SUM(line_amount)       AS return_spend
    FROM r
    GROUP BY customer_id
),
cty AS (
    SELECT customer_id, arg_max(country, invoice_date) AS country
    FROM p
    GROUP BY customer_id
),
future_buyers AS (
    SELECT t.customer_id
    FROM transactions_clean t, params
    WHERE t.invoice_date >= params.cutoff
      AND t.invoice_date <  params.cutoff + 90
    GROUP BY t.customer_id
    HAVING SUM(t.line_amount) > 0
)
SELECT
    params.cutoff                                              AS snapshot_date,
    a.customer_id,
    -- RFM
    date_diff('day', a.last_date,  params.cutoff)              AS recency_days,
    date_diff('day', a.first_date, params.cutoff)              AS tenure_days,
    a.n_orders,
    a.n_purchase_days,
    a.n_products,
    a.total_spend,
    a.total_spend / a.n_orders                                 AS avg_order_value,
    -- finestre recenti e trend
    a.orders_30d, a.orders_60d, a.orders_90d,
    a.spend_30d,  a.spend_60d,  a.spend_90d,
    a.orders_prev_90d,
    a.spend_prev_90d,
    a.orders_90d - a.orders_prev_90d                           AS orders_trend_90d,
    a.spend_90d  - a.spend_prev_90d                            AS spend_trend_90d,
    -- regolarità: NULL se il cliente ha comprato in un solo giorno
    g.avg_gap_days,
    g.std_gap_days,
    date_diff('day', a.last_date, params.cutoff) / NULLIF(g.avg_gap_days, 0) AS recency_over_gap,
    -- resi
    COALESCE(rt.n_return_invoices, 0)                          AS n_return_invoices,
    COALESCE(rt.return_spend, 0) / a.total_spend               AS return_ratio,
    -- paese
    CAST(c.country = 'United Kingdom' AS INTEGER)              AS is_uk,
    -- spesa netta e tassi normalizzati
    GREATEST(a.total_spend - COALESCE(rt.return_spend, 0), 0)  AS net_spend,
    a.n_orders / (GREATEST(date_diff('day', a.first_date, params.cutoff), 30) / 30.0) AS orders_per_30d,
    a.orders_180d,
    a.spend_180d,
    -- B2B vs B2C
    a.avg_qty_per_line,
    a.avg_unit_price,
    -- trend normalizzati e ultimo gap
    a.spend_90d / (a.spend_prev_90d + 1.0)                     AS spend_ratio_90d,
    a.spend_90d / NULLIF(a.total_spend, 0)                     AS spend_share_90d,
    g.last_gap_days,
    -- etichetta
    CASE WHEN fb.customer_id IS NULL THEN 1 ELSE 0 END         AS churn
FROM agg a
CROSS JOIN params
LEFT JOIN gap_stats g      ON g.customer_id  = a.customer_id
LEFT JOIN ret rt           ON rt.customer_id = a.customer_id
LEFT JOIN cty c            ON c.customer_id  = a.customer_id
LEFT JOIN future_buyers fb ON fb.customer_id = a.customer_id
WHERE date_diff('day', a.last_date, params.cutoff) <= 180
ORDER BY a.customer_id