# IPL First Innings Score Prediction

Predicts IPL first innings scores with scikit-learn regression models, and serves them through a Streamlit web app.

## Features

- Decision Tree, Random Forest and AdaBoost regressors
- Model selection in the app sidebar, with error metrics for the selected model
- Interactive charts built with Plotly

## Files

- `dataset/ipl.csv`: match data
- `ipl_prediction.py`: training script
- `streamlit_app.py`: web app
- `run_ipl_prediction.bat`: Windows launcher

## Run

```sh
pip install streamlit pandas numpy plotly scikit-learn
streamlit run streamlit_app.py
```
