import plotly.express as px

def market_summary(df):
    by_year = df.groupby("year", as_index=False)["selling_price"].median()
    fig = px.line(by_year, x="year", y="selling_price", markers=True,
                  title="Median listed price by manufacturing year",
                  labels={"year":"Manufacturing year","selling_price":"Median price (INR)"})
    fig.update_layout(template="plotly_white", margin=dict(l=15,r=15,t=55,b=15))
    return fig

def brand_summary(df):
    return (df.groupby("brand", as_index=False)
            .agg(vehicles=("selling_price","count"),
                 median_price=("selling_price","median"),
                 average_km=("km_driven","mean"))
            .sort_values("median_price", ascending=False)
            .round(0))
