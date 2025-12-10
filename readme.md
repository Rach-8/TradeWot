
When looking at the whole past data of SPY, I noticed that market was in either sideways pattern or in a trending pattern.
Then started developing 2 stretegies.

One which would prioritize saving capital in a choppy sideways market. 
Other would prioritize on making positive trades in an upward trending market.

modularized the code to make sure that I could work on these strategies individually and that
changing one, wouldnt affect the other.

Trending Strategy : Out performs the market in all upward trending scenarios.
                    Uses bollinger bands and 200 sma.
                    Simple but consistent.

Sideways Strategy : Gets the same returns as the market in most scenarios, in some scenarios even 
                    makes a trade to outperform market by a couple percent.
                    Complex and Very conserative.


Now the most important part would be to be able to predict if the market is sideways or trending.
For that, I will be training a model and giving it past 30 years of market targets, expecting it 
to predict sideways or trending. And then, that specific strategy will be run based on its 
reccomendation.