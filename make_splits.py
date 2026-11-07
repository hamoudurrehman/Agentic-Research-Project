import json, random, collections
d = json.load(open(r"data/bird/dev/dev_20240627/dev.json"))
rng = random.Random(42)
by = collections.defaultdict(list)
for x in d:
    by[x["db_id"]].append(x)
tune, ev = [], []
for db, items in by.items():
    rng.shuffle(items)
    k = max(1, round(len(items) * 0.25))
    tune += items[:k]
    ev += items[k:]
rng.shuffle(tune); rng.shuffle(ev)
json.dump(tune, open("data/bird/split_tune.json", "w"), indent=1)
json.dump(ev, open("data/bird/split_eval.json", "w"), indent=1)
print(len(tune), len(ev))