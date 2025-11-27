# 1D Wrap-Board Gomoku: Official Rules & Logic

## 1. Core Concept & Topology
[cite_start]Unlike Standard Gomoku, which is played on a Euclidean plane where edges act as hard boundaries, **1D Wrap-Board Gomoku** is played on a continuous linear vector[cite: 9].

* [cite_start]**The Board:** While visually represented as a $10 \times 10$ grid (Width $W=10$, Height $H=10$), the logical board is a single 1D array of 100 addressable positions (Indices $0$ to $99$)[cite: 37, 40].
* **The "Lie" of the Grid:** The 2D grid is merely a rasterization of the linear data. [cite_start]The "glitch" of a horizontal line wrapping from the end of one row to the beginning of the next is the central game mechanic[cite: 9, 23].
* **Topological Shape:** The board functions effectively as a **Cylinder** (or helix). [cite_start]The right edge of row $k$ is mathematically connected to the left edge of row $k+1$[cite: 56].

## 2. Winning Condition
[cite_start]The objective remains to align **five stones** of the same color continuously[cite: 7]. [cite_start]However, "continuity" is defined by **Arithmetic Stride** rather than visual adjacency[cite: 46].

[cite_start]A winning line is a sequence of indices $\{a_0, a_1, a_2, a_3, a_4\}$ such that the difference between adjacent indices is a constant Stride $S$[cite: 47].

## 3. Movement & Strides (Adjacency)
[cite_start]On a visual board of Width $W=10$, the valid connections (Strides) are calculated as follows[cite: 49]:

| Direction | Visual Vector | Logic: Stride Value ($S$) | Behavior |
| :--- | :--- | :--- | :--- |
| **Horizontal** | $(0, 1)$ | **+1** | Moves to the immediate next index. |
| **Vertical** | $(1, 0)$ | **+10** | Adds the board width ($W$) to the index. |
| **Major Diagonal** | $(1, 1)$ | **+11** | Adds $W + 1$. Moves down and right. |
| **Minor Diagonal** | $(1, -1)$ | **+9** | Adds $W - 1$. Moves down and left. |

## 4. The Rules of "Wrapping"
The defining feature of this variant is how lines interact with the visual "edges" of the board.

### 4.1. Horizontal Wrap (Stride 1)
* [cite_start]**Rule:** The end of a row is contiguous with the start of the next row[cite: 90].
* **Example:** A line can exist at indices $8, 9, 10, 11$.
* [cite_start]**Visual Effect:** The line exits the right edge of Row 1 and immediately re-enters on the left edge of Row 2[cite: 56].
* [cite_start]**Strategic Implication:** The edges are not defensive walls; a horizontal line cannot be blocked simply by running it into the frame of the board[cite: 95].

### 4.2. Minor Diagonal Wrap (Stride 9)
* [cite_start]**Rule:** A diagonal step of $+9$ connects the visual "left" and "right" sides of the board, but with a vertical shear[cite: 113].
* [cite_start]**The Anomaly:** A step of $+9$ from the left edge of a row (e.g., Index 40 at Row 5, Col 1) lands on the right edge of the *same* row (Index 49 at Row 5, Col 10) [cite: 58-60].
* [cite_start]**Visual Effect:** This creates a "teleporter" effect between the extreme columns[cite: 112].

### 4.3. Major Diagonal Wrap (Stride 11)
* [cite_start]**Rule:** A diagonal step of $+11$ performs a "Knight's Move" shift at the boundary[cite: 71].
* [cite_start]**Example:** From Index 9 (Row 1, Col 10), adding 11 results in Index 20 (Row 3, Col 1) [cite: 68-69].
* [cite_start]**Visual Effect:** The line exits the right edge of Row 1 and skips Row 2 entirely to re-enter on the left edge of Row 3[cite: 70].

### 4.4. Vertical Stability (No Wrap)
* [cite_start]**Rule:** Vertical lines (Stride 10) do **not** wrap horizontally[cite: 120].
* [cite_start]**Boundary:** Because the board is a Linear Strip (finite cylinder) and not a Torus, vertical lines terminate strictly at the top (Index 0-9) and bottom (Index 90-99) of the grid [cite: 121-122].

## 5. Strategic Guidelines
1.  **Count, Don't Look:** Do not rely on visual lines, as the grid is deceptive. [cite_start]Calculate indices to confirm connections (e.g., if you have stones at 34 and 45, you have a Stride 11 connection) [cite: 215-216].
2.  **The Seam:** Columns 1 and 10 constitute the "Seam" of the cylinder. [cite_start]Control here allows attacks to transition between rows [cite: 175-176].
3.  [cite_start]**Center Dominance:** The center of the 1D array (Indices 40-60) is the optimal opening zone as it maximizes extension potential in all directions without hitting the true Top/Bottom boundaries [cite: 188-189].

## 6. Summary Comparison

| Feature | Standard Gomoku | 1D Wrap-Board Gomoku |
| :--- | :--- | :--- |
| **Grid Logic** | Euclidean Plane ($15 \times 15$) | [cite_start]Linear Strip ($1 \times 100$) [cite: 225] |
| **Edge Behavior** | Hard boundaries (Blocks lines) | [cite_start]Illusions (Lines wrap/continue) [cite: 171] |
| **Victory Check** | 2D Adjacency | [cite_start]Arithmetic Sequence [cite: 225] |
| **Winning Line** | 5 contiguous visual stones | [cite_start]5 stones with constant index difference [cite: 47] |