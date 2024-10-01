import streamlit as st
from transformers import pipeline

model = 'vazish/mobile_bert_autofill'

with st.spinner('Loading the model...'):
    pipe = pipeline(
        'text-classification',
        model=model,
        truncation=True,
        batch_size=8,
        device=0
    )

st.success("Model loaded!")


input = st.text_area(
    "Enter HTML tags to predict separated by ',' and Press 'Classify'",
    '<input autocomplete="new-password" class="_1dHn9" data-hook="wsr-input" maxlength="524288" style="text-overflow: clip;" type="password" value=""/>,\n\n'
    '<input autocomplete="on" class="" data-v-0933c420="" id="undefined_billing_zip" name="billing_zip" step="1" type="text" value=""/>,\n\n'
    '<li><a data-analytics-menu-title="Rio de Janeiro" href="https://www.hotel-bb.com/de/stadt/hotels-rio-de-janeiro">Rio de Janeiro</a></li>',
    height=250
)

if st.button("Classify..."):
    st.write(pipe(input.split(',\n')))
