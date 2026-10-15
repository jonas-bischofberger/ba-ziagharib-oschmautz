from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import pyarrow.parquet as pq
import streamlit as st

S3_URI = "s3://vfb-collected/pff_fc_free_dataset_wc_2022/preprocessed/"
DATA_DIR = Path(__file__).parent / "data"

st.set_page_config(page_title="PFF World Cup 2022", layout="wide")
st.title("PFF FC World Cup 2022")

files = sorted(DATA_DIR.rglob("*.parquet"))
if not files:
    st.warning(f"No data found in {DATA_DIR}")
    st.stop()

# Read the first
st.header("Inspecting a *.parquet file")
n_rows_rows = 100
choice = st.selectbox("File", [str(f.relative_to(DATA_DIR)) for f in files])
st.dataframe(next(pq.ParquetFile(DATA_DIR / choice).iter_batches(batch_size=n_rows_rows)).to_pandas())

st.header("Inspecting event-tracking frames")
df_meta = pd.read_parquet(DATA_DIR / "metadata.parquet")
games = {int(r["id"]): f"{r['homeTeam.name']} – {r['awayTeam.name']} ({r['date'][:10]})" for _, r in df_meta.iterrows()}
available_games = [int(f.stem) for f in (DATA_DIR / "events_tracking").glob("*.parquet")]
game_id = st.selectbox("Game", sorted(available_games), format_func=lambda g: games.get(g, str(g)))


@st.cache_data
def load_events_tracking(game_id):
    return pd.read_parquet(DATA_DIR / "events_tracking" / f"{game_id}.parquet")


df = load_events_tracking(game_id)
df["frame_id"] = df.groupby(["gameEventId", "possessionEventId", "eventTime"], dropna=False).ngroup()
df_events = df.drop_duplicates("frame_id").set_index("frame_id")

# Pass end = ball position at the next (strictly later) event; the data has no end coordinates
df_ball_xy = df[df["tracking.team"] == "ball"].drop_duplicates("frame_id").set_index("frame_id")[["tracking.x", "tracking.y"]]
df_events_by_time = df_events.sort_values("eventTime")
nxt = df_events_by_time["eventTime"].searchsorted(df_events["eventTime"], side="right")
next_frame = pd.Series(df_events_by_time.index.append(pd.Index([-1]))[nxt], index=df_events.index)
is_pass = df_events["possessionEvents.possessionEventType"] == "PA"
df_events["pass_end_x"] = next_frame[is_pass].map(df_ball_xy["tracking.x"])
df_events["pass_end_y"] = next_frame[is_pass].map(df_ball_xy["tracking.y"])


def label(f):
    e = df_events.loc[f]
    return (f"{e['gameEvents.startFormattedGameClock']} P{e['gameEvents.period']} · "
            f"{e['gameEvents.gameEventType']} / {e['possessionEvents.possessionEventType']} · {e['gameEvents.playerName']}")


frame = st.select_slider("frame_id", options=df_events.index.tolist(), format_func=label)
fr = df[df["frame_id"] == frame]
game = df_meta.set_index("id").loc[str(game_id)]
length, width = fr["stadiumMetadata.pitchLength"].iloc[0], fr["stadiumMetadata.pitchWidth"].iloc[0]

# Plotting
fig = go.Figure()
L, W = length / 2, width / 2
for x0, x1, y0, y1 in [(-L, L, -W, W), (-L, -L + 16.5, -20.16, 20.16), (L - 16.5, L, -20.16, 20.16),
                       (-L, -L + 5.5, -9.16, 9.16), (L - 5.5, L, -9.16, 9.16)]:
    fig.add_shape(type="rect", x0=x0, x1=x1, y0=y0, y1=y1, line=dict(color="white"))
fig.add_shape(type="line", x0=0, x1=0, y0=-W, y1=W, line=dict(color="white"))
fig.add_shape(type="circle", x0=-9.15, x1=9.15, y0=-9.15, y1=9.15, line=dict(color="white"))

for side in ["home", "away"]:
    t = fr[fr["tracking.team"] == side]
    fig.add_scatter(
        x=t["tracking.x"], y=t["tracking.y"], mode="markers+text", name=game[f"{side}Team.name"],
        text=t["tracking.jerseyNum"].astype("Int64").astype(str), textfont=dict(color=game[f"{side}TeamKit.primaryTextColor"]),
        marker=dict(size=22, color=game[f"{side}TeamKit.primaryColor"], line=dict(color="black", width=1)),
        hovertext=t["tracking.positionGroupType"],
    )
ball = fr[fr["tracking.team"] == "ball"]
fig.add_scatter(x=ball["tracking.x"], y=ball["tracking.y"], mode="markers", name="Ball",
                marker=dict(size=10, color="yellow", line=dict(color="black", width=1)))
e = df_events.loc[frame]
if len(ball) and pd.notna(e["pass_end_x"]):
    fig.add_annotation(x=e["pass_end_x"], y=e["pass_end_y"],
                       ax=ball["tracking.x"].iloc[0], ay=ball["tracking.y"].iloc[0], xref="x", yref="y", axref="x", ayref="y",
                       arrowhead=3, arrowwidth=2, arrowcolor="yellow", standoff=6, text="")
fig.update_layout(plot_bgcolor="#2e7d32", height=650, xaxis=dict(range=[-L - 3, L + 3], showgrid=False, zeroline=False),
                  yaxis=dict(range=[-W - 3, W + 3], showgrid=False, zeroline=False, scaleanchor="x"))
st.plotly_chart(fig, width="stretch")
st.dataframe(df_events.loc[[frame]].T.dropna().astype(str), height=400)
