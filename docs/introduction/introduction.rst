===============================
``Qlib``: Quantitative Platform
===============================

Introduction
===================

.. image:: ../_static/img/logo/white_bg_rec+word.png
    :align: center

``Qlib`` is an AI-oriented quantitative investment platform, which aims to realize the potential, empower the research, and create the value of AI technologies in quantitative investment.

With ``Qlib``, users can easily try their ideas to create better Quant investment strategies.

Framework
===================
   
.. image:: ../_static/img/framework.png
    :align: center


At the module level, Qlib is a platform that consists of above components. The components are designed as loose-coupled modules and each component could be used stand-alone.

======================  ==============================================================================
Name                    Description
======================  ==============================================================================
`Data layer`            `DataServer` focuses on providing high-performance infrastructure for users to
                        manage and retrieve raw data. `DataEnhancement` will preprocess the data and 
                        provide the best dataset to be fed into the models.

`Interday Model`        `Interday model` focuses on producing prediction scores (aka. `alpha`). Models
                        are trained by `Model Creator` and managed by `Model Manager`. Users could 
                        choose one or multiple models for prediction. Multiple models could be combined
                        with `Ensemble` module.

`Interday Strategy`     `Portfolio Generator` will take prediction scores as input and output the 
                        orders based on the current position to achieve the target portfolio.

`Intraday Trading`      `Order Executor` is responsible for executing orders output by 
                        `Interday Strategy` and returning the executed results.

`Analysis`              Users could get a detailed analysis report of forecasting signals and portfolios
                        in this part.
======================  ==============================================================================

- The modules with hand-drawn style are under development and will be released in the future.
- The modules with dashed borders are highly user-customizable and extendible.
