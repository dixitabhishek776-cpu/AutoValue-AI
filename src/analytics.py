import plotly.express as px


def market_summary(df):
    by_year = (
        df.groupby("year", as_index=False)["price"]
        .median()
        .sort_values("year")
    )

    fig = px.line(
        by_year,
        x="year",
        y="price",
        markers=True,
        title="Median listed price by manufacturing year",
        labels={
            "year": "Manufacturing year",
            "price": "Median listed price (INR)",
        },
    )

    fig.update_layout(
        template="plotly_white",
        margin=dict(l=15, r=15, t=55, b=15),
    )

    return fig


def brand_summary(df):
    return (
        df.groupby("brand", as_index=False)
        .agg(
            listings=("price", "count"),
            median_price=("price", "median"),
            average_km=("distance_travelled(kms)", "mean"),
        )
        .sort_values("median_price", ascending=False)
        .round(0)
    )


def city_summary(df):
    return (
        df.groupby("city", as_index=False)
        .agg(
            listings=("price", "count"),
            median_price=("price", "median"),
        )
        .sort_values("listings", ascending=False)
        .round(0)
    )


def fuel_summary(df):
    return (
        df.groupby("fuel_type", as_index=False)
        .agg(
            listings=("price", "count"),
            median_price=("price", "median"),
        )
        .sort_values("listings", ascending=False)
        .round(0)
    )
