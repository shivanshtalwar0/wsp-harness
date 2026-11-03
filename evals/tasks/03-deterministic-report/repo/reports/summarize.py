"""Team spend report."""


def summarize(rows):
    """One row per team: ``{"team", "total", "n"}``.

    Sorted by total descending, ties by team name ascending.
    """
    groups = {}
    for row in rows:
        groups.setdefault(row["team"], []).append(row["score"])

    out = [{"team": team, "total": round(sum(scores), 2), "n": len(scores)}
           for team, scores in groups.items()]
    out.sort(key=lambda r: str(r["total"]), reverse=True)
    return out
