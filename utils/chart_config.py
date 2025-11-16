"""
Chart Configuration Module
Professional chart styling for Stock Analyzer platform
"""

def get_chart_config():
    """
    Returns standard Plotly configuration for all charts
    Implements the new professional color palette
    """
    return {
        'font': {
            'family': 'Inter, -apple-system, BlinkMacSystemFont, sans-serif',
            'size': 13,
            'color': '#C8C4C9'  # text-secondary
        },
        'plot_bgcolor': '#1E1B18',  # bg-primary
        'paper_bgcolor': '#1E1B18',  # bg-primary
        'template': 'plotly_dark',
        'xaxis': {
            'gridcolor': '#626C66',  # border-subtle
            'gridwidth': 1,
            'showline': False,
            'zeroline': False,
            'tickfont': {
                'family': 'Inter, sans-serif',
                'size': 12,
                'color': '#9A969B'  # text-tertiary
            }
        },
        'yaxis': {
            'gridcolor': '#626C66',  # border-subtle
            'gridwidth': 1,
            'showline': False,
            'zeroline': True,
            'zerolinecolor': '#626C66',
            'zerolinewidth': 1,
            'tickformat': ',.0f',
            'tickfont': {
                'family': 'JetBrains Mono, monospace',
                'size': 12,
                'color': '#9A969B'  # text-tertiary
            }
        },
        'hoverlabel': {
            'bgcolor': '#2A2622',  # bg-secondary
            'bordercolor': '#7A8479',  # border-default
            'font': {
                'family': 'JetBrains Mono, monospace',
                'size': 13,
                'color': '#FFFAFF'  # text-primary
            }
        },
        'showlegend': True,
        'legend': {
            'orientation': 'h',
            'yanchor': 'bottom',
            'y': 1.02,
            'xanchor': 'right',
            'x': 1,
            'font': {
                'family': 'Inter, sans-serif',
                'size': 13,
                'color': '#C8C4C9'
            },
            'bgcolor': 'rgba(42, 38, 34, 0.8)',
            'bordercolor': '#626C66',
            'borderwidth': 1
        }
    }


def get_semantic_colors():
    """
    Returns semantic color mapping for charts
    """
    return {
        'positive': '#7FC7B7',      # Sage Teal
        'negative': '#3B020A',      # Deep Burgundy (for backgrounds/fills)
        'negative_text': '#B91C28',  # Lightened Burgundy (for text/lines)
        'warning': '#C89F5F',       # Warm Gold
        'info': '#7FC7B7',          # Sage Teal
        'neutral': '#626C66',       # Cool Sage Gray
        'accent': '#7FC7B7',        # Primary Accent
        'secondary': '#C89F5F'      # Secondary Accent
    }


def apply_chart_theme(fig, title=None):
    """
    Apply professional theme to a Plotly figure

    Args:
        fig: Plotly figure object
        title: Optional chart title

    Returns:
        Updated figure with professional styling
    """
    config = get_chart_config()

    layout_update = {
        'font': config['font'],
        'plot_bgcolor': config['plot_bgcolor'],
        'paper_bgcolor': config['paper_bgcolor'],
        'template': config['template'],
        'xaxis': config['xaxis'],
        'yaxis': config['yaxis'],
        'hoverlabel': config['hoverlabel'],
        'showlegend': config['showlegend'],
        'legend': config['legend']
    }

    if title:
        layout_update['title'] = {
            'text': title,
            'x': 0.5,
            'xanchor': 'center',
            'font': {
                'family': 'Inter, -apple-system, BlinkMacSystemFont, sans-serif',
                'size': 18,
                'weight': 600,
                'color': '#FFFAFF'  # text-primary
            }
        }

    fig.update_layout(**layout_update)

    return fig


def get_color_scale_positive_negative():
    """
    Returns a color scale for positive/negative values
    """
    return [
        [0, '#3B020A'],      # Deep Burgundy (most negative)
        [0.5, '#626C66'],    # Neutral Gray
        [1, '#7FC7B7']       # Sage Teal (most positive)
    ]


def create_bar_colors(values):
    """
    Create color list for bar chart based on values

    Args:
        values: List of numeric values

    Returns:
        List of colors corresponding to values
    """
    colors = get_semantic_colors()
    return [colors['positive'] if v > 0 else colors['negative'] if v < 0 else colors['neutral'] for v in values]
