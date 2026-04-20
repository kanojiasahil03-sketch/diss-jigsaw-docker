# Alya0 Baseline — First Run Observations
Date: April 2026
Test image: 24pieces1.png (24-piece dinosaur puzzle)

## What I did
I cloned the Alya0 Jigsaw Puzzle Solver repository and ran it on the
provided sample image without modifying any code. The solver completed
in 1.97 seconds which was faster than I expected.

## What I observed
When I opened the output image, my first reaction was surprise. Some
pieces were connected correctly to their neighbours, but the overall
arrangement was wrong. It felt like the solver knew how to fit two
pieces together locally but had no idea where that pair should go on
the board. The sorting logic clearly struggled — pieces ended up in
the wrong regions entirely.

There were also visible gaps where pieces were either missing or placed
completely out of position. For a solver that has access to both the
piece geometry and a hint image, I expected much better results. The
fact that it partially failed even with a hint tells me the rule-based
approach has real limitations when it comes to global placement decisions.

## Pipeline
1. edge_extraction() — uses cv.findContours to detect each puzzle piece
   from the image, finds the 4 corners of each piece, and classifies
   every edge as either head (convex bump), hole (concave slot), or
   straight (border edge).

2. get_classified_pieces() — takes the border data and organises pieces
   into groups: corner pieces, edge pieces, and inner pieces. Corner and
   edge pieces are placed first because their straight edges make them
   easier to position.

3. order_pieces() — the rule-based placement engine. It uses three
   factors to decide where each piece goes: geometric edge matching
   (head fits hole), colour correlation between adjacent pieces, and
   SIFT feature matching against the hint image when one is provided.

## What becomes my RL observation space
The pieces_borders output from edge_extraction() gives me the edge
classification for each side of each piece (head/hole/straight). Combined
with the current board state — which positions are already filled — this
forms the observation vector my PPO agent will receive at every step.

## What my RL agent replaces
The order_pieces() function is the rule-based placement decision I am
replacing. Instead of hardcoded geometric rules, my PPO agent will learn
a policy that looks at the whole board and decides where to place the
next piece based on experience from thousands of training episodes.

## Observed limitations
- Gaps and misplacements clearly visible in the output image
- The solver matched pieces locally but got the global order wrong
- Breaks down on ambiguous edges where multiple pieces could fit
- Requires a hint image to perform reasonably — without it, results
  would likely be worse
- Optimised only for square-shaped pieces, not irregular shapes

## Test result
- 24pieces1.png solved in 1.97 seconds
- Partial success — some pieces placed correctly, others in wrong
  positions or missing entirely
- This confirms the baseline has clear room for improvement, which
  motivates the RL approach in my dissertation
