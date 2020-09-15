# backtest

modules

simulate true trading environment


- Order
- Exchange
- Position
- Account
- Report

backtest demo
    
    auto-update cross different modules from trade order

strategy framework

## Order

trade order
- Order.SELL: sell order, default 0
- Order.BUY: buy order, default 1
- direction: `Order.SELL` for sell, `Order.BUY` for buy
- sotck_id
- amount
- trade_date : pd.Timestamp

## Exchange

the stock exanchge, deal the trade order, provide stock market information

### Exchange Property
- trade_dates : list of pd.Timestamp
- codes : list stock_id list
- deal_price : str, 'close', 'open', 'vwap'
- quote : dataframe by D.features, trading data cache, default None
- limit_threshold : float, 0.1 for example, default None
- open_cost : cost rate for open, default 0.0015
- close_cost : cost rate for close, default 0.0025
- min_cost : min transaction cost, default 5
- trade_unit : trade unit, 100 for China A market

### Exchange Function
- check_stock_limit : buy limit, True for cannot trade, limit_threshold
- check_stock_suspended : check if suspended
- check_order : check is executable, include limit and suspend
- deal_order : (order, trade_account=None, position=None),if the order id executable, return trade_val, trade_cost, trade_price
- get price information realated, in this way need to check suspend first, (stock_id, trade_date)
    - get_close
    - get_deal_price
- generate_amount_position_from_weight_position : for strategy use
- generate_order_for_target_amount_position : generate order_list from target_position ( {stock_id : amount} ) and current_position({stock_id : amount})
- calculate_amount_position_value : value
- compare function : compare position dict

## Position

state of asset

including cash and stock

for each stock, contain 
- count : holding days
- amount : stock amount
- price : stock price

### Functions:
- update_order
    - buy_stock
    - sell_stock
- update postion information
    - cash
    - price
    - amount
    - count
- calculate value : use price in postion to calculate value
    - calculate_stock_value : without cash
    - calculate_value : with cash
- get information
    - get_stock_list
    - get_stock_price
    - get_stock_amount
    - get_stock_count
    - get_cash
- add_count_all : add 1 to all stock count
- transform
    - get_stock_amount_dict
    - get_stock_weight_dict : use price in postion to calculate value

## Report

daily report for account

- account postion value for each trade date
- daily return rate for each trade date
- turnover for each trade date
- trade cost for each trade date
- value for each trade date
- cash
- latest_report_date : pd.TimeStamp
    
### Function
- is_empty
- get_latest_date
- get_latest_account_value
- update_report_record
- generate_report_dataframe

## Account

state for the stock_trader

- curent position : Position() class
- trading related
    - return
    - turnover
    - cost
    - earning
- postion value
    - val 
    - cash
- report : Report()
- today

### Funtions

- get 
    - get_positions
    - get_cash
- init_state
- update_order(order, trade_val, cost) : update current postion and trading metrix after the order is dealed
- update_daily_end() : when the end of trade date, summarize today 
    - update rtn , from order's view
    - update price for each stock still in current position
    - update value for this account
    - update earning (2nd view of return , position' view)
    - update holding day, count of stock
    - update position hitory
    - update report
    

## backtest_demo

trade strategy:
    
    parameters : 
        topk : int, select topk stocks
        buffer_margin : size of buffer margin
        
    description :
        hold topk stocks at each trade date
        when adjust position
            the score model will generate scores for each stock
            if the stock of current position not in top buffer_margin score, sell them out;
            then equally buy recommended stocks
    
    the previous version of this strategy is in evaluate.py
    
    demo.py accomplishes same trading strategy with modules of Order, Exchange, Position, Report and Account
    
    test_strategy_demo.py did the consistency check between evaluate.py and demo.py
    
    strategy.py provide a strategy framework to do the backtest 

## Strategy

strategy framework

    strategy will generate orders if given pred_scores and market environment information
    there are two stages:
    1. generate target position
    2. generate order from target postion and current position

document for the framework

    the document shows some examples to accomplish those two stages

two strategy demo: 
- Strategy_amount_demo
- Strategy_weight_demo

backtest_demo with using strategy
