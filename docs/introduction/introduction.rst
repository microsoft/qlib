===============================
``Qlib``: Quantitative Platform
===============================

Introduction
===================

``Qlib`` is an AI-oriented quantitative investment platform, which aims to realize the potential, empower the research, and create the value of AI technologies in quantitative investment.

With ``Qlib``, users can easily try their ideas to create better Quant investment strategies.

Framework
==================

.. image:: ../_static/img/framework.png
    :alt: Framework


At the module level, Qlib is a platform that consists of above components. The components are designed as loose-coupled modules and each component could be used stand-alone.

======================  ========================================================================
Name                    Description
======================  ========================================================================

`Data layer`            `DataServer` focus on providing high performance infrastructure for users
                        to manage and retrieve raw data. `DataEnhancement` will preprocess the data
                        and provide the best dataset to be fed into the models.

`Interday Model`        `Interday model` focus on producing forecasting signals(aka. `alpha`).
                        Models are trained by `Model Creator` and managed by `Model Manager`.
                        Users could choose one or multiple models for forecasting. Multiple 
                        models could be combined with `Ensemble` module

`Interday Strategy`     `Portfolio Generator` will take forecasting signals as input and output 
                        the orders based on current position to achieve target portfolio                  
                        
`Intraday Trading`      `Order Executor` is responsible for executing orders output by 
                        `Interday Strategy` and returning the executed results.

`Analysis`              Users could get detailed analysis report of forecasting signal and portfolio
                        in this part.
======================  ========================================================================

- The modules with hand-drawn style is under development and will be released in the future.
- The modules with dashed border is highly user-customizable and extendible.
