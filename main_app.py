from dashboard import dashboard

app = dashboard.build_app()

if __name__ == '__main__':
    app.run_server(debug=True)
    # app.run_server(debug ='False', port = 8080, host ='0.0.0.0')