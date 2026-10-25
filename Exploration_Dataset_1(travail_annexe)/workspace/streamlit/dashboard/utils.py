import streamlit.components.v1 as components

def mermaid_viewer(code: str, height=650):
    html = f"""
    <script src="https://cdn.jsdelivr.net/npm/mermaid/dist/mermaid.min.js"></script>

    <style>
      .toolbar {{
        display: flex;
        gap: 10px;
        margin-bottom: 10px;
      }}
      .toolbar button {{
        padding: 6px 12px;
        background-color: #0057b8; /* Couleur produit */
        color: white;
        border: none;
        border-radius: 4px;
        cursor: pointer;
        font-size: 14px;
      }}
      .toolbar button:hover {{
        background-color: #003f82;
      }}
      #mermaid-container {{
        transform-origin: 0 0;
        transition: transform 0.2s ease;
      }}
    </style>

    <div class="toolbar">
      <button onclick="zoomIn()">Zoom +</button>
      <button onclick="zoomOut()">Zoom -</button>
      <button onclick="resetZoom()">Reset</button>
    </div>

    <div id="mermaid-container" class="mermaid">
      {code}
    </div>

    <script>
      mermaid.initialize({{
        startOnLoad: true,
        theme: "base",
        themeVariables: {{
          primaryColor: "#0057b8",
          primaryTextColor: "#ffffff",
          primaryBorderColor: "#003f82",
          lineColor: "#003f82",
          secondaryColor: "#e8f0fe",
          tertiaryColor: "#f5f9ff",
          fontFamily: "Inter, Arial"
        }}
      }});

      let scale = 1;

      function zoomIn() {{
        scale += 0.1;
        document.getElementById("mermaid-container").style.transform = "scale(" + scale + ")";
      }}

      function zoomOut() {{
        scale = Math.max(0.2, scale - 0.1);
        document.getElementById("mermaid-container").style.transform = "scale(" + scale + ")";
      }}

      function resetZoom() {{
        scale = 1;
        document.getElementById("mermaid-container").style.transform = "scale(1)";
      }}
    </script>
    """

    components.html(html, height=height, scrolling=True)
