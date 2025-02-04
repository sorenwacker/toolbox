from __future__ import annotations
import os
import fnmatch
import json
from pathlib import Path
from typing import List, Optional, Set, Dict, Any
import dash
from dash import html, dcc
import dash_cytoscape as cyto
from dash.dependencies import Input, Output, State
from flask import Flask

cyto.load_extra_layouts()

class DirectoryGraphVisualizer:
    """Interactive visualization of directory structures using Dash and Cytoscape."""
    
    LAYOUTS = ['breadthfirst', 'circle', 'cola', 'concentric', 'cose', 'cose-bilkent', 
               'dagre', 'euler', 'grid', 'klay', 'preset', 'random', 'spread']
    
    DIRECTORY_COLORS = ['#4CAF50', '#FFA500', '#2196F3', '#9C27B0', '#F44336', '#00BCD4', '#8BC34A']

    def __init__(self, root_directory: str | Path, 
                 port: int = 8050, 
                 ignore_patterns: List[str] = None,
                 max_depth: Optional[int] = None, 
                 include_files: bool = False):
        """Initialize the visualizer with the given parameters."""
        self.root_directory = Path(root_directory)
        if not self.root_directory.exists() or not self.root_directory.is_dir():
            raise ValueError(f"Invalid directory: {root_directory}")
            
        self.port = port
        self.ignore_patterns = set(ignore_patterns or [])
        self.max_depth = max_depth
        self.include_files = include_files
        
        self.server = Flask(__name__)
        self.app = dash.Dash(__name__, server=self.server)
        self.app.title = self.root_directory.name
        self._setup_layout()
        self._setup_callbacks()

    def _should_ignore(self, path: Path) -> bool:
        return any(fnmatch.fnmatch(path.name, pattern) for pattern in self.ignore_patterns)

    def _create_directory_elements(self) -> List[Dict[str, Any]]:
        elements = []
        root_id = str(self.root_directory).replace('\\', '/')

        def walk(path: Path, depth: int = 0) -> None:
            print(path)
            if self._should_ignore(path):
                return

            try:
                entries = [e for e in path.iterdir() if not self._should_ignore(e)]
            except (PermissionError, OSError) as e:
                print(f"Error accessing {path}: {e}")
                return

            current_id = str(path).replace('\\', '/')
            node_color = self.DIRECTORY_COLORS[depth % len(self.DIRECTORY_COLORS)]
            
            elements.append({
                'data': {
                    'id': current_id,
                    'label': path.name or str(path),
                    'is_root': current_id == root_id,
                    'depth': depth,
                    'type': 'directory'
                },
                'style': {'background-color': node_color}
            })

            if self.max_depth is not None and depth >= self.max_depth:
                return

            for entry in entries:
                entry_id = str(entry).replace('\\', '/')
                
                if entry.is_dir():
                    elements.append({'data': {'source': current_id, 'target': entry_id}})
                    walk(entry, depth + 1)
                elif self.include_files:
                    elements.append({
                        'data': {
                            'id': entry_id,
                            'label': entry.name,
                            'depth': depth + 1,
                            'type': 'file'
                        }
                    })
                    elements.append({'data': {'source': current_id, 'target': entry_id}})

        walk(self.root_directory)
        return elements

    def _get_stylesheet(self, edge_width: float = 1) -> List[Dict[str, Any]]:
        return [
            {
                'selector': 'node',
                'style': {
                    'content': 'data(label)',
                    'text-valign': 'center',
                    'text-halign': 'center',
                    'font-size': '12px',
                    'text-wrap': 'wrap',
                    'text-max-width': '80px',
                }
            },
            {
                'selector': 'node[is_root]',
                'style': {
                    'border-width': '1px',
                    'border-color': '#000000'
                }
            },
            {
                'selector': 'node[type="directory"]',
                'style': {'shape': 'ellipse'}
            },
            {
                'selector': 'node[type="file"]',
                'style': {
                    'shape': 'rectangle',
                    'background-color': '#4CAF50'
                }
            },
            {
                'selector': 'edge',
                'style': {
                    'width': edge_width,
                    'line-color': '#ccc',
                    'curve-style': 'bezier'
                }
            }
        ]

    def _setup_layout(self) -> None:
        layout_options = [{'label': layout, 'value': layout} for layout in self.LAYOUTS]
        
        self.app.layout = html.Div([
            html.H1(self.root_directory.name),  # Heading displays the directory name
            html.Div([
                dcc.Dropdown(id='layout-dropdown', options=layout_options, value='klay', 
                            style={'width': '300px', 'marginBottom': '10px'}),
                dcc.Slider(id='edge-width-slider', min=1, max=10, step=0.1, value=1,
                        marks={i: str(i) for i in range(1, 11)},
                        tooltip={"placement": "bottom", "always_visible": True}),
                html.Button("Remove Labels", id="toggle-labels-button", n_clicks=0),
                html.Button("Export to HTML", id="export-button"),
                html.Div(id='export-message')
            ], style={'margin': '20px'}),
            cyto.Cytoscape(
                id='directory-graph',
                layout={'name': 'cola'},
                style={'width': '100%', 'height': '1200px'},
                elements=[],
                stylesheet=self._get_stylesheet()
            )
        ])


    def _setup_callbacks(self) -> None:
        @self.app.callback(
            Output('directory-graph', 'elements'),
            Input('directory-graph', 'elements')
        )
        def update_elements(elements):
            return self._create_directory_elements() if not elements else elements

        @self.app.callback(
            Output('directory-graph', 'layout'),
            Input('layout-dropdown', 'value')
        )
        def update_layout(layout_name):
            return {'name': layout_name, 'roots': f'[id = "{str(self.root_directory).replace("\\", "/")}"]'} \
                if layout_name == 'breadthfirst' else {'name': layout_name}

        @self.app.callback(
            Output('directory-graph', 'stylesheet'),
            Input('edge-width-slider', 'value'),
            Input('toggle-labels-button', 'n_clicks')
        )
        def update_stylesheet(edge_width, n_clicks):
            stylesheet = self._get_stylesheet(edge_width)
            if n_clicks % 2 == 1:  # Toggle label visibility on odd clicks
                for style in stylesheet:
                    if style['selector'] == 'node':
                        style['style']['content'] = ''  # Hide labels
            return stylesheet

        @self.app.callback(
            Output('export-message', 'children'),
            Input('export-button', 'n_clicks'),
            State('directory-graph', 'elements'),
            prevent_initial_call=True
        )
        def export_visualization(n_clicks, elements):
            if n_clicks:
                try:
                    self._save_as_html(elements)
                    return "Exported to directory_visualization.html"
                except Exception as e:
                    return f"Export failed: {e}"
            return ""


    def _save_as_html(self, elements: List[Dict[str, Any]], filename: str = 'directory_visualization.html') -> None:
        html_template = """
        <!DOCTYPE html>
        <html><head><title>Directory Visualization</title>
        <script src="https://cdnjs.cloudflare.com/ajax/libs/cytoscape/3.19.0/cytoscape.min.js"></script>
        <style>body{margin:0}#cy{width:100vw;height:100vh}</style>
        </head><body><div id="cy"></div><script>
        cytoscape({
            container: document.getElementById('cy'),
            elements: %s,
            style: %s,
            layout: {name:'breadthfirst',roots:'[id = "%s"]'}
        });
        </script></body></html>
        """ % (json.dumps(elements), json.dumps(self._get_stylesheet()), 
               str(self.root_directory).replace('\\', '/'))
        
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(html_template)

    def run(self, debug: bool = True) -> None:
        self.app.run_server(debug=debug, port=self.port)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Visualize directory structure")
    parser.add_argument("directory", help="Root directory to visualize")
    parser.add_argument("--port", type=int, default=8050)
    parser.add_argument("--ignore", nargs="+", help="Patterns to ignore")
    parser.add_argument("--max-depth", type=int)
    parser.add_argument("--include-files", action="store_true")
    args = parser.parse_args()
    
    visualizer = DirectoryGraphVisualizer(
        args.directory, 
        port=args.port,
        ignore_patterns=args.ignore,
        max_depth=args.max_depth,
        include_files=args.include_files
    )
    visualizer.run()