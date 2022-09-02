mkdir -p ~/.streamlit/

cat > ~/.streamlit/config.toml <<EOF
[server]
headless = true
port = ${PORT:-8042}
enableCORS = true
EOF