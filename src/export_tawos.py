from __future__ import annotations

import argparse

import pandas as pd

ID_CANDS = ["ID", "Id", "Issue_ID"]
TITLE_CANDS = ["Title", "Summary", "Issue_Title"]
DESC_CANDS = ["Description_Text", "Description"]
DATE_CANDS = ["Creation_Date", "Created", "Created_Date", "Creation_Time"]
SP_CANDS = ["Story_Point", "Story_Points", "StoryPoint"]
PROJ_FK_CANDS = ["Project_ID", "ProjectID", "Project_Id"]
PROJ_NAME_CANDS = ["Name", "Project_Name", "Title"]


def pick(cols, cands, what, table):
    for c in cands:
        if c in cols:
            return c
    raise SystemExit(f"Could not find the {what} column in table {table}. Columns present: {list(cols)}\n"
                     f"Add the right name to the candidate list near the top of src/export_tawos.py.")


def build(issue: pd.DataFrame, project: pd.DataFrame) -> pd.DataFrame:
    print("Issue columns  :", list(issue.columns))
    print("Project columns:", list(project.columns))
    i_id = pick(issue.columns, ID_CANDS, "issue id", "Issue")
    i_title = pick(issue.columns, TITLE_CANDS, "title", "Issue")
    i_desc = pick(issue.columns, DESC_CANDS, "description", "Issue")
    i_date = pick(issue.columns, DATE_CANDS, "creation date", "Issue")
    i_sp = pick(issue.columns, SP_CANDS, "story point", "Issue")
    i_fk = pick(issue.columns, PROJ_FK_CANDS, "project id", "Issue")
    p_id = pick(project.columns, ID_CANDS, "project id", "Project")
    p_name = pick(project.columns, PROJ_NAME_CANDS, "project name", "Project")
    print(f"Using: id={i_id} title={i_title} desc={i_desc} date={i_date} sp={i_sp} fk={i_fk} | project id={p_id} name={p_name}")
    issue = issue[issue[i_sp].notna()]
    proj = project[[p_id, p_name]].rename(columns={p_id: "_pid", p_name: "Project_Name"})
    out = issue.rename(columns={i_fk: "_pid"}).merge(proj, on="_pid", how="left")
    out = out.rename(columns={i_id: "Issue_ID", i_title: "Title", i_desc: "Description_Text",
                              i_date: "Creation_Date", i_sp: "Story_Point"})
    return out[["Issue_ID", "Project_Name", "Title", "Description_Text", "Creation_Date", "Story_Point"]]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mysql", action="store_true")
    ap.add_argument("--db", default="tawos")
    ap.add_argument("--host", default="localhost")
    ap.add_argument("--user", default="root")
    ap.add_argument("--password", default="")
    ap.add_argument("--csv-dir")
    ap.add_argument("--out", default="data/tawos_story_points.csv")
    a = ap.parse_args()
    if a.mysql:
        import pymysql
        con = pymysql.connect(host=a.host, user=a.user, password=a.password, database=a.db)
        issue = pd.read_sql("SELECT * FROM Issue WHERE Story_Point IS NOT NULL", con)
        project = pd.read_sql("SELECT * FROM Project", con)
    elif a.csv_dir:
        issue = pd.read_csv(f"{a.csv_dir}/Issue.csv")
        project = pd.read_csv(f"{a.csv_dir}/Project.csv")
    else:
        raise SystemExit("choose --mysql or --csv-dir")
    out = build(issue, project)
    out.to_csv(a.out, index=False)
    print(f"wrote {len(out)} issues with story points from {out.Project_Name.nunique()} projects -> {a.out}")


if __name__ == "__main__":
    main()
