import unittest


class TestCacheOrder(unittest.TestCase):
    def test_cache_order(self):
        from atp_sim_sdk_py import replenish_scene, strategy, entity, rolling_env

        st = strategy.StrategyB()
        scene = replenish_scene.CacheOrder(st)
        roller = rolling_env.RollingEnv(scene, rts_day=4)
        sku = entity.SKU(10, 1, 2, 3, lead_time=2, yesterday_end_stock=15)

        roller.reset(sku)

        datasets = {
            "multipliers": [1.0] * 10,
            "predicts": [10.1, 10, 11, 12, 9] * (2 + 1),
            "sales": [8, 7, 12, 9, 50, 8, 7, 8, 9, 8],
        }

        for i, multiplier in enumerate(datasets["multipliers"]):
            multiplier = 1.0
            predicts = datasets["predicts"][i : i + 5]  # must lead_time<len
            sales = datasets["sales"][i]

            roller.rolling(sku, multiplier, predicts, sales)

            # sku.show_snapshot()
            print(f"sku: {sku.snapshot_result()} {sales} {predicts}")
            sku.snapshot_result()
            roller._snapshot()
        roller._snapshot_table()
        roller._snapshot_wait()
        print(f"done: {roller.summary_result()}")


TestCacheOrder().test_cache_order()
