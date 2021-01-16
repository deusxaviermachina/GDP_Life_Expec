import numpy as np
import pandas as pd

def scrape():
    url = "https://worldpopulationreview.com/countries/countries-by-gdp"
    dataset1=pd.read_html(url)
    url="https://worldpopulationreview.com/countries/life-expectancy"
    dataset2=pd.read_html(url)
    df=pd.DataFrame(dataset1[0], columns=["Name","GDP Per Capita"])
    df=df.dropna()
    Y=[]
    for i in df["GDP Per Capita"]:
        i=float(i.replace("$","").replace(",",""))
        Y.append(i)
    df2=pd.DataFrame(Y, columns=["GDP Per Capita(usd)"])
    merged=df.join(df2["GDP Per Capita(usd)"])
    del merged["GDP Per Capita"]
    df2=pd.DataFrame(dataset2[0], columns=["Name","Total"])
    df2=df2.dropna()
    merged=merged.dropna()
    #pd.set_option("display.max_rows", None)
    joined=pd.merge(merged, df2, on="Name", how='left')
    #print(joined)
    data=joined.fillna(np.mean(df2["Total"]))
    data=data.drop(index=[210,207,204,201,200,198,176,181,188,194])
    return data