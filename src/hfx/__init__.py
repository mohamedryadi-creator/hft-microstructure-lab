"""hfx -- high-frequency microstructure: price formation, market design, RL.

Every piece of mathematics lives in a plain, importable, unit-tested function.
Notebooks orchestrate and plot; they never derive.

Sub-packages
------------
``hfx.itch``
    Nasdaq TotalView-ITCH 5.0: the wire format, a streaming decoder, and a
    synthetic encoder that gives the decoder something to be checked against.
``hfx.book``
    Order-by-order book reconstruction, queue tracking, trade classification.
``hfx.hawkes``
    Simulation, maximum likelihood, the time-rescaling goodness-of-fit test, and
    the closed-form signature plot of the mutually exciting price model.
``hfx.vol``
    Realized volatility under microstructure noise -- two-scale, pre-averaged,
    kernel -- and the uncertainty-zones model of a price on a tick grid.
``hfx.queue``
    The queue-reactive model: intensity estimation, invariant law, simulator.
``hfx.mm``
    Market making: the Guéant-Lehalle-Fernandez-Tapia closed form, its
    calibration, and a reinforcement learner benchmarked against it.
``hfx.design``
    Make-take fees as a principal-agent problem between exchange and market
    maker.
``hfx.pipeline``
    The symbol panel, the extraction from the raw feed, and the study that turns
    it into the committed measurements.
``hfx.viz``
    Shared matplotlib styling.
"""

__version__ = "0.1.0"

__all__ = ["__version__"]
