
import os, json, pandas as pd

try:
    import great_expectations as ge
    HAVE_GE = True
except Exception:
    HAVE_GE = False

DATA_DIR = os.environ.get("DATA_DIR","./data")
EXP_DIR = os.environ.get("EXPECT_DIR","./ml/expectations")
OUT_DIR = os.environ.get("VALIDATION_OUT","./validation")

os.makedirs(OUT_DIR, exist_ok=True)

def load_json(p):
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)

def fallback_checks(df, spec, name):
    issues = []
    for c in spec.get("non_null", []):
        if c in df.columns and df[c].isna().sum() > 0:
            issues.append(f"{name}: {c} has {int(df[c].isna().sum())} nulls")
    for c in spec.get("positive", []):
        if c in df.columns:
            neg = (pd.to_numeric(df[c], errors="coerce") < 0).sum()
            if neg > 0:
                issues.append(f"{name}: {c} has {int(neg)} negative values")
    for rel in spec.get("relations", []):
        if len(rel)==3 and rel[1]=="approx_eq" and rel[2]=="price*days":
            if set(["price","days","total"]).issubset(df.columns):
                calc = pd.to_numeric(df["price"], errors="coerce") * pd.to_numeric(df["days"], errors="coerce")
                total = pd.to_numeric(df["total"], errors="coerce")
                diff = (calc - total).abs()
                if (diff > 1e-6).sum() > 0:
                    issues.append(f"{name}: total != price*days for {int((diff>1e-6).sum())} rows")
    return issues

def run():
    flights = pd.read_csv(os.path.join(DATA_DIR, "flights.csv"))
    hotels  = pd.read_csv(os.path.join(DATA_DIR, "hotels.csv"))
    fexp = load_json(os.path.join(EXP_DIR, "flights_expectations.json"))
    hexp = load_json(os.path.join(EXP_DIR, "hotels_expectations.json"))

    report = {"engine": "great_expectations" if HAVE_GE else "fallback", "results": {}}

    if HAVE_GE:
        f_df = ge.from_pandas(flights)
        for c in fexp.get("non_null", []):
            f_df.expect_column_values_to_not_be_null(c)
        for c in fexp.get("positive", []):
            if c in flights.columns:
                f_df.expect_column_values_to_be_between(c, min_value=0)
        f_res = f_df.validate()
        report["results"]["flights"] = {"success": f_res["success"]}
    else:
        issues = fallback_checks(flights, fexp, "flights")
        report["results"]["flights"] = {"success": len(issues)==0, "issues": issues}

    if HAVE_GE:
        h_df = ge.from_pandas(hotels)
        for c in hexp.get("non_null", []):
            h_df.expect_column_values_to_not_be_null(c)
        for c in hexp.get("positive", []):
            if c in hotels.columns:
                h_df.expect_column_values_to_be_between(c, min_value=0)
        h_res = h_df.validate()
        report["results"]["hotels"] = {"success": h_res["success"]}
    else:
        issues = fallback_checks(hotels, hexp, "hotels")
        report["results"]["hotels"] = {"success": len(issues)==0, "issues": issues}

    out_path = os.path.join(OUT_DIR, "validation_report.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(out_path)

if __name__ == "__main__":
    run()
