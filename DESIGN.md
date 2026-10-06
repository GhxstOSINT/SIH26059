# Southern Passage interface

The interface opens with an original orbital scene and scroll-directed visual transition into the operator workspace. The visual language is cinematic: near-black space, detailed polar imagery, large uppercase Space Grotesk, fine telemetry labels, and restrained white controls. The map remains the first operational figure; source dates and review boundaries stay visible.

## Type and color

- Space Grotesk 300–700: brand and display headings. DM Sans 400–700: controls and body copy. DM Mono: dates, measurements, and coordinate data only.
- Background `#030812`; surface `#071320`; primary text `#f6f8f7`; secondary text `#b7c9cf`; active `#e9f7f7`; ice emphasis `#e4c594`.
- Preserve the contrast of source status, disabled controls, and observational warnings. Do not use color alone to communicate a route or review state.

## Interaction

- The opening image pans and scales with scroll. Headline phases crossfade and the scene resolves into the workspace; a direct workspace link bypasses this sequence. Honor reduced-motion preferences.
- Work-area navigation has a clear active underline. The route map, observed data, and review governance retain separate claims and purposes.
- Dense source details, route assumptions, and historical track information remain in disclosures; they must be keyboard accessible.
- The app is HTML/CSS/JavaScript without a React runtime. The orbital image is generated specifically for Southern Passage and is labeled as a concept visual, never an observation.

## Sources

Visual and motion direction: [Edolus](https://edolus.com/). The sequence and imagery are original implementations, not copied site assets or source. Navigation icon paths were adapted from Lucide via the [Better Icons](https://github.com/better-auth/better-icons) CLI (MIT-licensed).
