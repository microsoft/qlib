===============================
``Qlib``: Quantitative Library
===============================

Introduction
===================

``Qlib`` is an AI-oriented quantitative investment platform, which aims to realize the potential, empower the research, and create the value of AI technologies in quantitative investment.

With ``Qlib``, users can easily apply their favorite model to create better Quant investment strategy.


Framework
==================

.. image:: ../_static/img/framework.png
    :alt: Framework


At module level, ``Qlib`` is a platform that consists of the above components. Each components is loose-coupling and can be used stand-alone.

======================  ========================================================================
Name                    Description
======================  ========================================================================
`Data layer`            `DataServer` focus on providing high performance infrastructure for user
                        to retrieve and get raw data. `DataEnhancement` will preprocess the data
                        and provide the best dataset to be fed in to the models.

`Interday Model`        `Interday model` focus on producing forecasting signals(aka. `alpha`). 
                        Models are trained by `Model Creator` and managed by `Model Manager`.
                        User could choose one or multiple models for forecasting. Multiple models
                        could be combined with `Ensemble` module.

`Interday Strategy`     `Portfolio Generator` will take forecasting signals as input and output 
                        the orders based on current position to achieve target portfolio.

`Intraday Trading`      `Order Executor` is responsible for executing orders output by 
                        `Interday Strategy` and returning the executed results.

`Analysis`              User could get detailed analysis report of forecasting signal and portfolio
                        in this part.
======================  ========================================================================

- The modules with hand-drawn style is under development and will be released in the future.
- The modules with dashed border is highly user-customizable and extendible.
