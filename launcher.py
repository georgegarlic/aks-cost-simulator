import os, sys

if getattr(sys, "frozen", False):
    base = sys._MEIPASS
else:
    base = os.path.dirname(os.path.abspath(__file__))
os.chdir(base)

import streamlit.web.bootstrap

sys.argv = ["streamlit", "run", "app.py"]
flag_options = {"server.port": os.environ.get("PORT", "8080")}
streamlit.web.bootstrap.run("app.py", "", [], flag_options)
