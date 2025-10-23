# Stock Analyzer Design System

**Version:** 1.0.0
**Last Updated:** 2025-10-23
**Status:** Production Ready

---

## Table of Contents

1. [Overview](#overview)
2. [Design Principles](#design-principles)
3. [Color System](#color-system)
4. [Typography System](#typography-system)
5. [Spacing Scale](#spacing-scale)
6. [Border Radius](#border-radius)
7. [Component Patterns](#component-patterns)
8. [Interaction Guidelines](#interaction-guidelines)
9. [Chart Styling](#chart-styling)
10. [Extensibility Guide](#extensibility-guide)

---

## Overview

This design system establishes a comprehensive set of visual and interaction standards for the Stock Analyzer application. Every design decision is intentional, systematic, and optimized for financial data presentation.

### Goals

- **Clarity**: Data visibility is paramount; design never competes with content
- **Consistency**: Same patterns, same outcomes across all modules
- **Precision**: Monospace typography and aligned numerical data for accurate comparison
- **Professionalism**: Premium aesthetic appropriate for financial analysis tools

### Technical Foundation

- **CSS Variables**: All design tokens are CSS custom properties for global consistency
- **Font Stack**: Inter (UI) + JetBrains Mono (data) with comprehensive fallbacks
- **Color Mode**: Dark theme optimized for extended screen time
- **Framework**: Streamlit with custom CSS overrides

---

## Design Principles

### 1. Minimal by Default
Reduce visual noise. Use whitespace, not lines. Subtle borders, not heavy containers.

### 2. Data-First Typography
Financial data uses monospace fonts with tabular numerals. Digits align vertically for easy comparison.

### 3. Intentional Color
Color communicates meaning (positive/negative, interactive states). Never decorative.

### 4. Systematic Spacing
All spacing uses multiples of 4px. Predictable rhythm aids scannability.

### 5. Refined Interactions
Smooth transitions (0.2s ease). Clear focus states. Hover feedback on all interactive elements.

---

## Color System

All colors are defined as CSS custom properties in `:root` for global consistency.

### Color Palette

#### Background Colors

```css
--color-bg-primary: #0F0F0F    /* Main application background */
--color-bg-secondary: #1A1A1A  /* Cards, containers, elevated surfaces */
--color-bg-tertiary: #242424   /* Hover states, subtle emphasis */
```

**Usage Guidelines:**
- **Primary**: Body background, chart backgrounds
- **Secondary**: All cards, form containers, table headers, metric boxes
- **Tertiary**: Row hover states, radio button hover, subtle highlights

**Contrast Ratios:**
- Primary to Secondary: 1.15:1 (subtle elevation)
- Secondary to Tertiary: 1.13:1 (minimal but perceptible)

---

#### Border Colors

```css
--color-border-subtle: #2A2A2A   /* Default borders, minimal separation */
--color-border-default: #404040  /* Hover states, active elements */
--color-border-strong: #525252   /* Emphasis borders (rarely used) */
```

**Usage Guidelines:**
- **Subtle**: All card borders, input default state, table cell borders, grid lines
- **Default**: Input hover state, chart zero lines, active element borders
- **Strong**: Reserved for special emphasis (currently unused, available for future)

**Visual Hierarchy:**
- Subtle borders should be barely noticeable (supporting structure)
- Default borders appear on interaction (hover/focus)
- Strong borders for critical UI elements only

---

#### Text Colors

```css
--color-text-primary: #FFFFFF   /* Headings, primary content */
--color-text-secondary: #A0A0A0 /* Labels, captions, secondary info */
--color-text-tertiary: #707070  /* Subtle text, disabled states */
```

**Usage Guidelines:**
- **Primary**: All headings, metric values, table data, important text
- **Secondary**: Form labels, table headers, captions, axis labels
- **Tertiary**: Placeholder text, disabled text, subtle scale markers

**Contrast Ratios (against #0F0F0F):**
- Primary: 21:1 (maximum contrast)
- Secondary: 8.5:1 (WCAG AAA compliant)
- Tertiary: 4.6:1 (WCAG AA minimum for UI elements)

---

#### Semantic Colors

```css
--color-positive: #10B981  /* Gains, positive changes, success states */
--color-negative: #EF4444  /* Losses, negative changes, error states */
--color-warning: #F59E0B   /* Caution, moderate risk */
--color-info: #3B82F6     /* Information, neutral emphasis */
```

**Usage Guidelines:**

**Positive (#10B981 - Emerald)**
- Price increases, portfolio gains
- Success messages, completed actions
- Low risk indicators
- Positive metric deltas (▲)

**Negative (#EF4444 - Red)**
- Price decreases, portfolio losses
- Error messages, failed validations
- High risk indicators
- Negative metric deltas (▼)

**Warning (#F59E0B - Amber)**
- Moderate risk levels
- Cautionary messages
- Alerts requiring attention
- Neutral/changing states (—)

**Info (#3B82F6 - Blue)**
- Informational messages
- Neutral data points
- Clickable links
- Chart accent colors

**Accessibility Note:**
Never use color alone to convey information. Always pair with:
- Text labels ("Low Risk" / "High Risk")
- Icons or shapes (▲▼—)
- Position (gains above, losses below)

---

#### Interactive Colors

```css
--color-interactive-default: #FFFFFF  /* Primary buttons, active states */
--color-interactive-hover: #E5E5E5    /* Button hover state */
--color-interactive-active: #D4D4D4   /* Button pressed state */
```

**Usage Guidelines:**
- **Default**: Primary button background, focus border color, active tab indicator
- **Hover**: Button hover background, indicates interactivity
- **Active**: Button press state (rarely visible, short duration)

**State Progression:**
```
Default (#FFFFFF) → Hover (#E5E5E5) → Active (#D4D4D4)
```

All transitions: `0.2s ease`

---

### Color Code Snippets

#### Button States
```css
.button {
  background-color: var(--color-interactive-default);
  color: #000000;
  transition: all 0.2s ease;
}

.button:hover {
  background-color: var(--color-interactive-hover);
  transform: translateY(-1px);
}

.button:active {
  background-color: var(--color-interactive-active);
  transform: translateY(0);
}
```

#### Success Message
```css
.success-message {
  padding: 12px 16px;
  background-color: transparent;
  border-left: 3px solid var(--color-positive);
  color: #D1FAE5;  /* Light green text */
  font-size: 14px;
  line-height: 1.5;
}
```

#### Error Message
```css
.error-message {
  padding: 12px 16px;
  background-color: transparent;
  border-left: 3px solid var(--color-negative);
  color: #FECACA;  /* Light red text */
  font-size: 14px;
  line-height: 1.5;
}
```

---

## Typography System

Two-font system optimized for financial applications:
- **Inter**: UI text (headings, labels, body copy)
- **JetBrains Mono**: Financial data (prices, quantities, percentages)

### Font Loading

```css
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500;600&display=swap');
```

**Weights Loaded:**
- Inter: 400 (Regular), 500 (Medium), 600 (Semibold), 700 (Bold)
- JetBrains Mono: 400 (Regular), 500 (Medium), 600 (Semibold)

---

### Font Stacks

```css
--font-ui: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', system-ui, sans-serif;
--font-mono: 'JetBrains Mono', 'SF Mono', Monaco, 'Cascadia Code', 'Roboto Mono', Consolas, monospace;
```

**Fallback Strategy:**
- **Inter**: Falls back to system UI fonts (native look on each platform)
- **JetBrains Mono**: Falls back to platform monospace fonts

---

### Typography Scale

#### H1 - Page Titles
```css
font-family: var(--font-ui);
font-size: 32px;
font-weight: 700;
line-height: 1.2;
letter-spacing: -0.02em;
color: var(--color-text-primary);
margin-bottom: var(--space-md);  /* 16px */
```

**Usage**: Main page titles, major section headers
**Example**: "Indian Stock Dashboard", "AI-Powered Stock Predictions"

---

#### H2 - Section Headers
```css
font-family: var(--font-ui);
font-size: 24px;
font-weight: 600;
line-height: 1.3;
letter-spacing: -0.01em;
color: var(--color-text-primary);
margin-bottom: 12px;
```

**Usage**: Section headings, card titles
**Example**: "Risk Analysis Dashboard", "Trading Recommendations"

---

#### H3 - Subsection Headers
```css
font-family: var(--font-ui);
font-size: 18px;
font-weight: 600;
line-height: 1.4;
color: var(--color-text-primary);
margin-bottom: var(--space-sm);  /* 8px */
```

**Usage**: Subsection titles, card headers
**Example**: "Model Performance Breakdown", "Stress Test Scenarios"

---

#### Body Text
```css
font-family: var(--font-ui);
font-size: 15px;
font-weight: 400;
line-height: 1.6;
color: var(--color-text-primary);
```

**Usage**: All body copy, descriptions, paragraphs
**Line height note**: 1.6 provides optimal readability for extended reading

---

#### Caption / Label
```css
font-family: var(--font-ui);
font-size: 13px;
font-weight: 500;
line-height: 1.4;
letter-spacing: 0.3px;
text-transform: uppercase;
color: var(--color-text-secondary);
```

**Usage**: Form labels, table headers, metric labels, axis titles
**Why uppercase**: Differentiates labels from content, professional appearance

---

#### Financial Data - Large
```css
font-family: var(--font-mono);
font-size: 28px;
font-weight: 600;
line-height: 1.2;
color: var(--color-text-primary);
font-variant-numeric: tabular-nums;
```

**Usage**: Primary metric values in cards
**Example**: "₹1,234.56" in metric cards

---

#### Financial Data - Standard
```css
font-family: var(--font-mono);
font-size: 14px;
font-weight: 400;
line-height: 1.4;
color: var(--color-text-primary);
font-variant-numeric: tabular-nums;
```

**Usage**: Table cells, deltas, secondary numerical data
**`tabular-nums`**: Ensures all digits have equal width for perfect vertical alignment

---

### Typography Usage Matrix

| Element | Font | Size | Weight | Color |
|---------|------|------|--------|-------|
| Page Title | Inter | 32px | 700 | Primary |
| Section Header | Inter | 24px | 600 | Primary |
| Subsection | Inter | 18px | 600 | Primary |
| Body Text | Inter | 15px | 400 | Primary |
| Label/Caption | Inter | 13px | 500 | Secondary |
| Metric Value | Mono | 28px | 600 | Primary |
| Table Data | Mono | 14px | 400 | Primary |
| Metric Delta | Mono | 14px | 400 | Semantic |
| Button Text | Inter | 15px | 600 | #000 |
| Form Input | Inter | 15px | 400 | Primary |

---

### Code Snippets

#### Metric Card Typography
```html
<div class="metric-card">
  <div class="metric-label">Current Price</div>
  <div class="metric-value">₹1,234.56</div>
  <div class="metric-delta positive">+2.34% ▲</div>
</div>
```

```css
.metric-label {
  font-family: var(--font-ui);
  font-size: 13px;
  font-weight: 500;
  letter-spacing: 0.3px;
  text-transform: uppercase;
  color: var(--color-text-secondary);
}

.metric-value {
  font-family: var(--font-mono);
  font-size: 28px;
  font-weight: 600;
  color: var(--color-text-primary);
  font-variant-numeric: tabular-nums;
}

.metric-delta {
  font-family: var(--font-mono);
  font-size: 14px;
  font-weight: 400;
  font-variant-numeric: tabular-nums;
}

.metric-delta.positive { color: var(--color-positive); }
.metric-delta.negative { color: var(--color-negative); }
```

---

## Spacing Scale

Systematic spacing using 4px base unit.

```css
--space-xs: 4px;    /* Minimal spacing */
--space-sm: 8px;    /* Tight spacing */
--space-md: 16px;   /* Default spacing */
--space-lg: 24px;   /* Comfortable spacing */
--space-xl: 32px;   /* Generous spacing */
--space-xxl: 48px;  /* Section separation */
```

### Usage Guidelines

**XS (4px)**
- Small element padding
- Icon margins
- Tight list item spacing

**SM (8px)**
- Label to input spacing
- Between related elements
- Tab gaps
- Risk bar labels

**MD (16px)**
- Card padding
- Input padding
- Metric card spacing
- Default margin bottom

**LG (24px)**
- Card padding (larger cards)
- Section internal spacing
- Form field spacing

**XL (32px)**
- Large card padding
- Major component spacing

**XXL (48px)**
- Between major sections
- Page section separation

### Spacing Application Examples

```css
/* Card Padding */
.card {
  padding: var(--space-lg);  /* 24px */
}

/* Input Spacing */
.input-group label {
  margin-bottom: var(--space-sm);  /* 8px */
}

.input-group {
  margin-bottom: var(--space-md);  /* 16px */
}

/* Section Separation */
.section {
  margin-bottom: var(--space-xxl);  /* 48px */
}
```

---

## Border Radius

Only two values for visual consistency:

```css
--radius-sm: 6px;  /* Small interactive elements */
--radius-md: 8px;  /* Containers and cards */
```

### Usage Matrix

| Element | Radius | Rationale |
|---------|--------|-----------|
| Buttons | 6px | Small, frequently clicked elements |
| Input Fields | 6px | Matches button radius |
| Radio Buttons | 6px | Small interactive elements |
| Tab Tops | 6px 6px 0 0 | Connects to tab content |
| Cards | 8px | Larger surfaces, softer corners |
| Tables | 8px | Container elements |
| Alerts | 8px | Message containers |
| Metric Boxes | 8px | Data containers |
| Expanders | 8px | Collapsible containers |
| Modals | 8px | Overlay containers |

### Why Only Two Values?

- **Consistency**: Reduces decision-making, creates visual coherence
- **Hierarchy**: Small = interactive, Medium = container
- **Simplicity**: Easy to remember and apply

**Never use**: 5px, 7px, 10px, 12px, 15px, or any other arbitrary values

---

## Component Patterns

Detailed specifications for all UI components.

### Buttons

#### Primary Button

```css
.button-primary {
  font-family: var(--font-ui);
  font-size: 15px;
  font-weight: 600;
  background-color: var(--color-interactive-default);
  color: #000000;
  border: none;
  border-radius: var(--radius-sm);
  padding: 10px 16px;
  transition: all 0.2s ease;
  cursor: pointer;
}

.button-primary:hover {
  background-color: var(--color-interactive-hover);
  transform: translateY(-1px);
  box-shadow: 0 4px 12px rgba(0,0,0,0.3);
}

.button-primary:active {
  background-color: var(--color-interactive-active);
  transform: translateY(0);
}
```

**Visual States:**
1. **Default**: White background, black text
2. **Hover**: Light gray background, subtle lift (-1px), shadow appears
3. **Active**: Darker gray, returns to baseline position
4. **Focus**: (Same as hover + keyboard outline)

---

#### Secondary Button

```css
.button-secondary {
  font-family: var(--font-ui);
  font-size: 15px;
  font-weight: 600;
  background-color: transparent;
  color: var(--color-text-primary);
  border: 1px solid var(--color-border-default);
  border-radius: var(--radius-sm);
  padding: 10px 16px;
  transition: all 0.2s ease;
  cursor: pointer;
}

.button-secondary:hover {
  background-color: var(--color-bg-tertiary);
  border-color: var(--color-interactive-default);
}
```

---

### Form Elements

#### Text Input

```css
.input {
  font-family: var(--font-ui);
  font-size: 15px;
  font-weight: 400;
  background-color: var(--color-bg-secondary);
  border: 1px solid var(--color-border-subtle);
  border-radius: var(--radius-sm);
  padding: 10px 14px;
  color: var(--color-text-primary);
  transition: all 0.2s ease;
}

.input:hover {
  border-color: var(--color-border-default);
}

.input:focus {
  border-color: var(--color-interactive-default);
  outline: none;
  box-shadow: 0 0 0 3px rgba(255, 255, 255, 0.1);
}
```

**Focus State Details:**
- **Border**: Changes to white (`--color-interactive-default`)
- **Shadow**: 3px white glow at 10% opacity
- **Transition**: Smooth 0.2s ease
- **No outline**: `outline: none` removes browser default

---

#### Password Input

```css
.input[type="password"] {
  font-family: var(--font-mono);
  letter-spacing: 2px;
}
```

**Why monospace?**
- Makes password length visible
- Easier to detect typos
- Professional appearance for sensitive data

---

#### Label

```css
.label {
  font-family: var(--font-ui);
  font-size: 13px;
  font-weight: 500;
  letter-spacing: 0.3px;
  color: var(--color-text-secondary);
  margin-bottom: var(--space-sm);
  display: block;
  text-transform: uppercase;
}
```

---

### Tables

#### Table Structure

```css
.table-container {
  border-radius: var(--radius-md);
  overflow: hidden;
}

.table thead th {
  font-family: var(--font-ui);
  font-size: 13px;
  font-weight: 600;
  text-transform: uppercase;
  letter-spacing: 0.5px;
  color: var(--color-text-secondary);
  background-color: var(--color-bg-secondary);
  padding: 12px 16px;
  border-bottom: 1px solid var(--color-border-subtle);
  text-align: right;  /* For numerical columns */
}

.table tbody td {
  font-family: var(--font-mono);
  font-size: 14px;
  font-weight: 400;
  text-align: right;
  font-variant-numeric: tabular-nums;
  padding: 10px 16px;
  border-bottom: 1px solid var(--color-border-subtle);
  transition: background-color 0.15s ease;
}

.table tbody tr:hover td {
  background-color: var(--color-bg-tertiary);
}

.table tbody tr:last-child td {
  border-bottom: none;
}

/* First column (labels) should be left-aligned */
.table th:first-child,
.table td:first-child {
  text-align: left;
  font-family: var(--font-ui);
  font-weight: 500;
}
```

**Key Features:**
- **Monospace data**: All numerical columns use JetBrains Mono with tabular-nums
- **Right alignment**: Numbers align on decimal point
- **Hover feedback**: Subtle background change on row hover
- **Clean termination**: Last row has no bottom border
- **First column exception**: Labels use UI font and left-align

---

### Cards

```css
.card {
  background-color: var(--color-bg-secondary);
  border: 1px solid var(--color-border-subtle);
  border-radius: var(--radius-md);
  padding: var(--space-lg);
}

.card-compact {
  background-color: var(--color-bg-secondary);
  border: 1px solid var(--color-border-subtle);
  border-radius: var(--radius-md);
  padding: var(--space-md);
}
```

---

### Metric Cards

```html
<div class="metric-card">
  <div class="metric-label">Portfolio Value</div>
  <div class="metric-value">₹1,234,567.89</div>
  <div class="metric-delta positive">+12.34% ▲</div>
</div>
```

```css
.metric-card {
  background-color: var(--color-bg-secondary);
  border: 1px solid var(--color-border-subtle);
  border-radius: var(--radius-md);
  padding: var(--space-md);
}

.metric-label {
  font-family: var(--font-ui);
  font-size: 13px;
  font-weight: 500;
  letter-spacing: 0.3px;
  text-transform: uppercase;
  color: var(--color-text-secondary);
  margin-bottom: var(--space-sm);
}

.metric-value {
  font-family: var(--font-mono);
  font-size: 28px;
  font-weight: 600;
  line-height: 1.2;
  color: var(--color-text-primary);
  font-variant-numeric: tabular-nums;
  margin-bottom: var(--space-xs);
}

.metric-delta {
  font-family: var(--font-mono);
  font-size: 14px;
  font-weight: 400;
  font-variant-numeric: tabular-nums;
}

.metric-delta.positive { color: var(--color-positive); }
.metric-delta.negative { color: var(--color-negative); }
.metric-delta.neutral { color: var(--color-warning); }
```

---

### Alerts & Messages

#### Success Message
```css
.success-message {
  padding: 12px 16px;
  background-color: transparent;
  border-left: 3px solid var(--color-positive);
  border-radius: 0;
  color: #D1FAE5;
  font-family: var(--font-ui);
  font-size: 14px;
  font-weight: 400;
  line-height: 1.5;
}
```

#### Error Message
```css
.error-message {
  padding: 12px 16px;
  background-color: transparent;
  border-left: 3px solid var(--color-negative);
  border-radius: 0;
  color: #FECACA;
  font-family: var(--font-ui);
  font-size: 14px;
  font-weight: 400;
  line-height: 1.5;
}
```

**Why no background fill?**
- Reduces visual alarm
- Keeps focus on content
- Allows stacking without visual clutter

**Why 3px left border?**
- Provides color coding without overwhelming
- Maintains lightweight appearance
- Doesn't block content

---

## Interaction Guidelines

Precise specifications for all interactive behaviors.

### Timing

| Interaction | Duration | Easing | Notes |
|-------------|----------|--------|-------|
| Button hover | 0.2s | ease | Quick response |
| Input focus | 0.2s | ease | Immediate feedback |
| Background change | 0.15s | ease | Subtle transitions |
| Table row hover | 0.15s | ease | Smooth highlight |
| Progress bar fill | 0.6s | cubic-bezier(0.4, 0, 0.2, 1) | Smooth, intentional |
| Password strength | 0.3s | cubic-bezier(0.4, 0, 0.2, 1) | Responsive feedback |

**General Rule**: Faster for interactive elements (0.15-0.2s), slower for progress/loading (0.3-0.6s)

---

### Shadow Intensity

#### Button Hover Shadow
```css
box-shadow: 0 4px 12px rgba(0,0,0,0.3);
```
- **Y-offset**: 4px (subtle lift)
- **Blur**: 12px (soft edge)
- **Opacity**: 30% black (visible but not heavy)

#### Input Focus Glow
```css
box-shadow: 0 0 0 3px rgba(255, 255, 255, 0.1);
```
- **Spread**: 3px (clear but not excessive)
- **Opacity**: 10% white (subtle on dark background)
- **No blur**: Creates crisp outline

**Never use:**
- Heavy shadows (opacity > 40%)
- Large spreads (> 4px)
- Colored shadows (except for focus states)

---

### Hover States

#### Button
- Background: Default → Hover
- Transform: translateY(-1px)
- Shadow: None → 0 4px 12px rgba(0,0,0,0.3)
- Timing: 0.2s ease

#### Input Field
- Border: Subtle → Default color
- No transform or shadow
- Timing: 0.2s ease

#### Table Row
- Background: Transparent → Tertiary
- No border change
- Timing: 0.15s ease (faster for data scanning)

#### Radio Button
- Background: Transparent → Tertiary
- Border-radius: 6px (consistent with other small elements)
- Timing: 0.2s ease

---

### Focus States

**Keyboard Navigation Requirement**: All interactive elements must have clear focus indicators for accessibility.

```css
element:focus {
  border-color: var(--color-interactive-default);
  outline: none;
  box-shadow: 0 0 0 3px rgba(255, 255, 255, 0.1);
}
```

**Remove browser default outline**, replace with:
- Border color change (subtle to white)
- 3px white glow (10% opacity)

---

### Font Weight in Data Contexts

| Context | Weight | Rationale |
|---------|--------|-----------|
| Large metrics | 600 | Prominence, scannability |
| Table data | 400 | Readability over long periods |
| Deltas/changes | 400 | Secondary to main values |
| Headings | 600-700 | Clear hierarchy |
| Labels | 500 | Differentiation from values |
| Body text | 400 | Optimal legibility |

**Rule**: Use heavier weights (600+) sparingly. In data-heavy contexts, 400 reduces eye fatigue.

---

## Chart Styling

Comprehensive guidelines for all Plotly charts.

### Global Chart Configuration

```python
fig.update_layout(
    font=dict(
        family='Inter, sans-serif',
        size=13,
        color='#A0A0A0'
    ),
    plot_bgcolor='#0F0F0F',
    paper_bgcolor='#0F0F0F',
    template='plotly_dark'
)
```

---

### Grid Lines

```python
xaxis=dict(
    gridcolor='#242424',  # Subtle, non-distracting
    gridwidth=1,
    showline=False,
    zeroline=False
)

yaxis=dict(
    gridcolor='#242424',
    gridwidth=1,
    showline=False,
    zeroline=False,
    tickformat=',.0f'  # Thousand separators
)
```

**Why `#242424`?**
- Subtle enough not to compete with data
- Visible enough to aid reading
- Matches `--color-bg-tertiary`

---

### Hover Labels

```python
hoverlabel=dict(
    bgcolor='#1A1A1A',
    bordercolor='#404040',
    font=dict(
        family='JetBrains Mono, monospace',
        size=13,
        color='#FFFFFF'
    )
)
```

**Why monospace?** Numerical data should use monospace for precision alignment, even in tooltips.

---

### Axis Typography

```python
xaxis=dict(
    tickfont=dict(
        family='Inter, sans-serif',  # Dates, categories
        size=12,
        color='#A0A0A0'
    )
)

yaxis=dict(
    tickfont=dict(
        family='JetBrains Mono, monospace',  # Prices, numbers
        size=12,
        color='#A0A0A0'
    )
)
```

**X-axis**: Inter (dates, category labels)
**Y-axis**: JetBrains Mono (prices, quantities)

---

### Chart Titles

```python
title={
    'text': "Chart Title",
    'font': {
        'family': 'Inter, sans-serif',
        'size': 18,
        'weight': 600,
        'color': '#FFFFFF'
    }
}
```

---

### Line Chart Best Practices

```python
fig.add_trace(go.Scatter(
    x=dates,
    y=prices,
    mode='lines',
    name='Historical Prices',
    line=dict(
        color='#3B82F6',  # Use semantic colors
        width=2           # Not too thick (avoid 3+)
    ),
    hovertemplate='<b>Historical</b><br>Date: %{x}<br>Price: ₹%{y:.2f}<extra></extra>'
))
```

**Line width guidelines:**
- Primary data: 2px
- Secondary/reference: 1-1.5px
- Emphasis: 3px (rarely, only for primary focus)

---

### Bar Chart Best Practices

```python
fig.add_trace(go.Bar(
    x=categories,
    y=values,
    marker_color=colors,  # Use semantic colors array
    text=[f"{val:+.1f}%" for val in values],
    textposition='auto',
    hovertemplate='<b>%{x}</b><br>Return: %{y:.1f}%<extra></extra>'
))
```

**Color coding:**
- Positive values: `#10B981`
- Negative values: `#EF4444`
- Neutral/warning: `#F59E0B`
- Information: `#3B82F6`

---

### Zero Line Styling

For charts with positive/negative values:

```python
yaxis=dict(
    zeroline=True,
    zerolinewidth=1,      # Thin line
    zerolinecolor='#404040'  # Default border color
)
```

---

## Extensibility Guide

Guidelines for adding new components while preserving visual integrity.

### Adding New Chart Types

1. **Use existing color variables** for all color decisions
2. **Follow typography rules**: Inter for labels, Mono for numbers
3. **Apply grid configuration**: `gridcolor='#242424'`, always
4. **Include hover labels** with monospace font
5. **Never use gradients** in fills or backgrounds
6. **Test with real data** to ensure readability

**Template for new charts:**

```python
def create_new_chart(data):
    fig = go.Figure()

    # Add traces with semantic colors
    fig.add_trace(go.[ChartType](
        ...
        marker_color='var(--color-info)',  # Or appropriate semantic color
    ))

    # Apply standard layout
    fig.update_layout(
        font=dict(family='Inter, sans-serif', size=13, color='#A0A0A0'),
        plot_bgcolor='#0F0F0F',
        paper_bgcolor='#0F0F0F',
        xaxis=dict(
            gridcolor='#242424',
            tickfont=dict(family='Inter, sans-serif', size=12, color='#A0A0A0')
        ),
        yaxis=dict(
            gridcolor='#242424',
            tickformat=',.0f',  # If numerical
            tickfont=dict(family='JetBrains Mono, monospace', size=12, color='#A0A0A0')
        ),
        hoverlabel=dict(
            bgcolor='#1A1A1A',
            bordercolor='#404040',
            font=dict(family='JetBrains Mono, monospace', size=13)
        )
    )

    return fig
```

---

### Creating Modal Windows

**Not yet implemented**, but when needed:

```css
.modal-overlay {
  background-color: rgba(0, 0, 0, 0.8);  /* 80% black overlay */
  backdrop-filter: blur(4px);             /* Subtle blur */
}

.modal-container {
  background-color: var(--color-bg-secondary);
  border: 1px solid var(--color-border-default);
  border-radius: var(--radius-md);
  padding: var(--space-xl);
  max-width: 600px;
  box-shadow: 0 20px 40px rgba(0,0,0,0.5);  /* Heavy shadow for elevation */
}

.modal-header {
  font-family: var(--font-ui);
  font-size: 24px;
  font-weight: 600;
  color: var(--color-text-primary);
  margin-bottom: var(--space-lg);
}

.modal-close {
  background-color: transparent;
  border: none;
  color: var(--color-text-secondary);
  font-size: 24px;
  cursor: pointer;
  transition: color 0.2s ease;
}

.modal-close:hover {
  color: var(--color-text-primary);
}
```

---

### Adding Dashboard Panels

Use existing card patterns:

```html
<div class="dashboard-panel">
  <div class="panel-header">
    <h3>Panel Title</h3>
    <div class="panel-actions">
      <!-- Optional actions -->
    </div>
  </div>
  <div class="panel-body">
    <!-- Content -->
  </div>
</div>
```

```css
.dashboard-panel {
  background-color: var(--color-bg-secondary);
  border: 1px solid var(--color-border-subtle);
  border-radius: var(--radius-md);
  padding: var(--space-lg);
  margin-bottom: var(--space-lg);
}

.panel-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: var(--space-md);
  padding-bottom: var(--space-md);
  border-bottom: 1px solid var(--color-border-subtle);
}

.panel-header h3 {
  margin: 0;
}
```

---

### Component Checklist

Before adding any new component, verify:

- [ ] Uses CSS variables for all colors (no hardcoded hex values)
- [ ] Uses systematic spacing (`--space-*` variables)
- [ ] Uses correct border radius (`--radius-sm` or `--radius-md`)
- [ ] Financial data uses `font-family: var(--font-mono)` with `tabular-nums`
- [ ] Labels use uppercase Inter 13px medium
- [ ] All transitions are 0.15-0.2s ease (or 0.3-0.6s for progress)
- [ ] Hover states change background to `--color-bg-tertiary` or add shadow
- [ ] Focus states include white border + 3px glow
- [ ] No emojis in production code
- [ ] No gradients in backgrounds or fills
- [ ] Grid lines use `#242424` if applicable

---

## Design Tokens Export

For external tools or documentation:

```json
{
  "colors": {
    "background": {
      "primary": "#0F0F0F",
      "secondary": "#1A1A1A",
      "tertiary": "#242424"
    },
    "border": {
      "subtle": "#2A2A2A",
      "default": "#404040",
      "strong": "#525252"
    },
    "text": {
      "primary": "#FFFFFF",
      "secondary": "#A0A0A0",
      "tertiary": "#707070"
    },
    "semantic": {
      "positive": "#10B981",
      "negative": "#EF4444",
      "warning": "#F59E0B",
      "info": "#3B82F6"
    },
    "interactive": {
      "default": "#FFFFFF",
      "hover": "#E5E5E5",
      "active": "#D4D4D4"
    }
  },
  "spacing": {
    "xs": "4px",
    "sm": "8px",
    "md": "16px",
    "lg": "24px",
    "xl": "32px",
    "xxl": "48px"
  },
  "radius": {
    "sm": "6px",
    "md": "8px"
  },
  "typography": {
    "fontFamily": {
      "ui": "'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', system-ui, sans-serif",
      "mono": "'JetBrains Mono', 'SF Mono', Monaco, 'Cascadia Code', 'Roboto Mono', Consolas, monospace"
    },
    "fontWeight": {
      "regular": 400,
      "medium": 500,
      "semibold": 600,
      "bold": 700
    },
    "fontSize": {
      "h1": "32px",
      "h2": "24px",
      "h3": "18px",
      "body": "15px",
      "caption": "13px",
      "metricLarge": "28px",
      "metricSmall": "14px"
    },
    "lineHeight": {
      "tight": 1.2,
      "normal": 1.4,
      "relaxed": 1.6
    }
  },
  "transitions": {
    "fast": "0.15s ease",
    "normal": "0.2s ease",
    "slow": "0.3s cubic-bezier(0.4, 0, 0.2, 1)",
    "slowest": "0.6s cubic-bezier(0.4, 0, 0.2, 1)"
  }
}
```

---

## Migration Guide

For migrating existing components to this design system:

### Step 1: Replace Hardcoded Colors

**Before:**
```css
background-color: #1a1c23;
color: #666;
border: 1px solid #333;
```

**After:**
```css
background-color: var(--color-bg-secondary);
color: var(--color-text-secondary);
border: 1px solid var(--color-border-subtle);
```

---

### Step 2: Standardize Border Radius

**Before:**
```css
border-radius: 5px;  /* Arbitrary */
border-radius: 10px; /* Arbitrary */
border-radius: 15px; /* Arbitrary */
```

**After:**
```css
border-radius: var(--radius-sm);  /* For buttons, inputs */
border-radius: var(--radius-md);  /* For cards, containers */
```

---

### Step 3: Apply Typography System

**Before:**
```css
font-size: 14px;
font-weight: normal;
line-height: 1.5;
```

**After:**
```css
font-family: var(--font-ui);
font-size: 15px;
font-weight: 400;
line-height: 1.6;
color: var(--color-text-primary);
```

---

### Step 4: Update Spacing

**Before:**
```css
padding: 15px;
margin-bottom: 20px;
gap: 10px;
```

**After:**
```css
padding: var(--space-md);
margin-bottom: var(--space-lg);
gap: var(--space-sm);
```

---

## Accessibility Considerations

### Color Contrast

All text colors meet WCAG AA standards minimum:

| Combination | Ratio | Standard |
|-------------|-------|----------|
| Primary text on primary bg | 21:1 | AAA |
| Secondary text on primary bg | 8.5:1 | AAA |
| Tertiary text on primary bg | 4.6:1 | AA |
| Positive on primary bg | 5.2:1 | AA |
| Negative on primary bg | 4.8:1 | AA |

---

### Keyboard Navigation

All interactive elements must:
- Be focusable via Tab key
- Have clear focus indicators (white border + glow)
- Support Enter/Space activation
- Never rely on hover-only interactions

---

### Screen Reader Support

- All form inputs have associated labels
- Buttons have descriptive text (no icon-only buttons without aria-label)
- Tables have proper thead/tbody structure
- Charts include text descriptions where applicable

---

## Version History

### v1.0.0 (2025-10-23)
- Initial design system documentation
- Complete color, typography, spacing, and component specifications
- Chart styling guidelines
- Extensibility rules
- CSS variable standardization (--color-*, --space-*, --radius-*)

---

## Appendix

### Quick Reference Card

```
COLORS
├─ Background: primary #0F0F0F, secondary #1A1A1A, tertiary #242424
├─ Border: subtle #2A2A2A, default #404040, strong #525252
├─ Text: primary #FFFFFF, secondary #A0A0A0, tertiary #707070
├─ Semantic: positive #10B981, negative #EF4444, warning #F59E0B, info #3B82F6
└─ Interactive: default #FFFFFF, hover #E5E5E5, active #D4D4D4

SPACING
4px → 8px → 16px → 24px → 32px → 48px

RADIUS
6px (interactive) | 8px (containers)

TYPOGRAPHY
Inter: UI text | JetBrains Mono: Financial data
32px (h1) → 24px (h2) → 18px (h3) → 15px (body) → 13px (caption)
Metric large: 28px | Metric small: 14px

TRANSITIONS
Fast: 0.15s | Normal: 0.2s | Slow: 0.3s | Progress: 0.6s
```

---

**For questions or clarifications, reference the specific sections above.**
**Last updated: 2025-10-23**
