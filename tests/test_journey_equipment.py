from types import SimpleNamespace
from games.fancraft.journey import Journey


def fixture(equipment, bag):
    j = object.__new__(Journey)
    calls = []
    j.snap = lambda: {"equipment": {"equipment": equipment}, "inventory": bag}
    j.count = lambda item: sum(row["quantity"] for row in bag if row["itemId"] == item)
    def equip(op, itemId, hand):
        calls.append((op, itemId, hand))
        equipment[hand + "_hand"] = {"itemId": itemId}
    j.gw = SimpleNamespace(call=equip)
    return j, calls


def test_saved_equipment_does_not_require_duplicate_items_in_bag():
    j, calls = fixture({"left_hand": {"itemId": "guild_sword"}, "right_hand": {"itemId": "guild_shield"}}, [])
    result = j.equip_items([("guild_sword", "left"), ("guild_shield", "right")])
    assert "guild_sword" in result and "guild_shield" in result
    assert calls == []


def test_only_the_missing_hand_is_equipped(monkeypatch):
    monkeypatch.setattr("games.fancraft.journey.time.sleep", lambda _: None)
    j, calls = fixture({"left_hand": {"itemId": "guild_sword"}}, [{"itemId": "guild_shield", "quantity": 1}])
    result = j.equip_items([("guild_sword", "left"), ("guild_shield", "right")])
    assert calls == [("equip", "guild_shield", "right")]
    assert "guild_sword" in result and "guild_shield" in result
