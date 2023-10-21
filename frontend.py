import streamlit as st
import pandas as pd
import numpy as np
import requests

# https://docs.streamlit.io/library/api-reference

st.title('Deer Locator')

# @st.cache_data
# def load_data(nrows):
#     data = pd.read_csv(DATA_URL, nrows=nrows)
#     lowercase = lambda x: str(x).lower()
#     data.rename(lowercase, axis='columns', inplace=True)
#     data[DATE_COLUMN] = pd.to_datetime(data[DATE_COLUMN])
#     return data

def multi_inference(files):
    url = "http://localhost:8000/uploadfiles/"
    multi_files = (('files', x) for x in files)  # need to convert to tuples for the requests library

    response = requests.post(url=url, files=multi_files)
    return response



data = st.file_uploader("upload a video", type=['png', 'jpg'], accept_multiple_files=True)


if len(data) > 0:
# if data is not None:
    # Create a text element and let the reader know the data is loading.
    data_load_state = st.text('Loading data...')
    
    st.write(data)
    
    res = multi_inference(data)

    # for debugging 
    st.subheader("Request POST Header - just for debugging")
    st.json(dict(res.request.headers))
    st.subheader("Response Status Code - just for debugging")
    st.info(f'Status Code: {res.status_code}')
    st.subheader("Response Header - just for debugging")
    st.json(dict(res.headers))
    st.subheader("Response Content - just for debugging")
    st.write(res.content)

    data_load_state.text("Done! (using st.cache_data)")

    
    
    
# hour_to_filter = st.slider('hour', 0, 23, 17)  # min: 0h, max: 23h, default: 17h
# filtered_data = data[data[DATE_COLUMN].dt.hour == hour_to_filter]
# st.subheader(f'Map of all pickups at {hour_to_filter}:00')
# st.map(filtered_data)