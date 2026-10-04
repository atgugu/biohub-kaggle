"""Sanity-check the official scorer on synthetic graphs with known answers."""
import sys
sys.path[:0] = ["/workspace/biohub/research/data_metric/metric_code/src"]
import polars as pl
import tracksdata as td
from tracking_cellmot.metrics import evaluate

SCALE = (1.625, 0.40625, 0.40625)

def graph(nodes, edges):
    g = td.graph.InMemoryGraph()
    for k in ("z", "y", "x"):
        g.add_node_attr_key(k, pl.Float64, -1.0)
    ids = g.bulk_add_nodes([dict(t=t, z=float(z), y=float(y), x=float(x)) for t, z, y, x in nodes])
    if edges:
        g.bulk_add_edges([{"source_id": ids[a], "target_id": ids[b]} for a, b in edges])
    return g

# GT: track 0->1->2 that divides at node 2 into 3 and 4, each continuing one frame (5, 6).
gt_nodes = [(0, 30, 100, 100), (1, 30, 100, 102), (2, 30, 100, 104),
            (3, 30, 90, 104), (3, 30, 110, 104), (4, 30, 90, 106), (4, 30, 110, 106)]
gt_edges = [(0, 1), (1, 2), (2, 3), (2, 4), (3, 5), (4, 6)]

def run(name, nodes, edges, expect):
    er = evaluate(graph(nodes, edges), graph(gt_nodes, gt_edges), scale=SCALE, max_distance=7.0)
    got = (er.edge_tp, er.edge_fp, er.edge_fn, er.division_tp, er.division_fp, er.division_fn)
    print(f"{'OK ' if got == expect else 'BAD'} {name}: edge TP/FP/FN + div TP/FP/FN = {got} (expected {expect})")
    return got == expect

if __name__ == "__main__":
    ok = True
    ok &= run("perfect copy", gt_nodes, gt_edges, (6, 0, 0, 1, 0, 0))
    ok &= run("one daughter missing (no fork)", gt_nodes, [(0, 1), (1, 2), (2, 3), (3, 5), (4, 6)], (5, 0, 1, 0, 0, 1))
    # dt=2 edge must be dropped by the scorer
    ok &= run("gap edge dt=2 ignored", gt_nodes, [(0, 2), (2, 3), (2, 4), (3, 5), (4, 6)], (4, 0, 2, 1, 0, 0))  # fork node itself matches the GT parent -> valid anchor
    # wrong partners after the split: 3->6 and 4->5
    ok &= run("swapped grandchildren", gt_nodes, [(0, 1), (1, 2), (2, 3), (2, 4), (3, 6), (4, 5)], (4, 2, 2, 1, 0, 0))
    # duplicate fork: extra spurious child of node 1 (non-dividing annotated GT node) -> division FP
    ok &= run("fork on non-dividing GT node", gt_nodes + [(2, 30, 120, 104)], gt_edges + [(1, 7)], (6, 1, 0, 1, 1, 0))
    sys.exit(0 if ok else 1)
