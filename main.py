app = dash.Dash(__name__)
server = app.server  # THIS is needed by Render

# your layout & callbacks here

if __name__ == "__main__":
    app.run_server(debug=True)
