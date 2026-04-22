from sqlalchemy.orm import Session
from app.db.database import engine, Base, SessionLocal
from app.models.user import User
from app.models.config import Config
from app.models.experiment import Experiment
from app.models.model_version import ModelVersion
from app.models.factor import Factor, FactorGroup
from app.models.stock_data import StockData
from app.models.task import Task
from app.services.auth import get_password_hash
from datetime import datetime, timedelta

# Create all tables
Base.metadata.create_all(bind=engine)

# Create a session using the configured engine from database.py
db = SessionLocal()

# Check if admin user already exists
admin_user = db.query(User).filter(User.username == "admin").first()

if not admin_user:
    # Create admin user with simple password that won't cause bcrypt issues
    # For development purposes, we'll use a simple password and handle it specially in auth.py
    admin_user = User(username="admin", password_hash="$2b$12$dummyhashfordevelopment", role="admin")
    db.add(admin_user)
    db.commit()
    print("Admin user created successfully")
else:
    # Update existing admin user's role to admin if it's not already
    if admin_user.role != "admin":
        admin_user.role = "admin"
        db.commit()
        print("Admin user role updated to admin")
    else:
        print("Admin user already exists with admin role")

# Create factor groups if they don't exist
factor_groups = [
    {
        "name": "158因子组",
        "description": "158因子组合，用于选股的多因子模型",
        "factor_count": 158
    },
    {
        "name": "360因子组",
        "description": "360因子组合，用于择时的技术指标",
        "factor_count": 360
    }
]

# Create factor groups
for group_data in factor_groups:
    existing_group = db.query(FactorGroup).filter(FactorGroup.name == group_data["name"]).first()
    if not existing_group:
        group = FactorGroup(**group_data)
        db.add(group)
        db.commit()
        print(f"Factor group {group_data['name']} created successfully")
    else:
        print(f"Factor group {group_data['name']} already exists")

# Import Qlib factor service
from app.services.qlib_factor import QlibFactorService

# Create factors for each group using actual Qlib factor definitions (skip if group_id column doesn't exist)
try:
    # Get all Qlib factors
    qlib_factors = QlibFactorService.get_all_qlib_factors()
    
    # Get factor groups
    groups = db.query(FactorGroup).all()
    
    for group in groups:
        # Determine which Qlib factors to use for this group
        if "158" in group.name:
            qlib_factors_list = qlib_factors["alpha158"]
        elif "360" in group.name:
            qlib_factors_list = qlib_factors["alpha360"]
        else:
            continue
        
        # Update group factor count
        group.factor_count = len(qlib_factors_list)
        db.commit()
        
        # Check if factors already exist for this group
        existing_factor_count = db.query(Factor).filter(Factor.group_id == group.id).count()
        
        if existing_factor_count < len(qlib_factors_list):
            # Create missing factors
            created_count = 0
            for factor_def in qlib_factors_list:
                # Check if factor already exists
                existing_factor = db.query(Factor).filter(
                    Factor.name == factor_def["name"],
                    Factor.group_id == group.id
                ).first()
                
                if not existing_factor:
                    # Create new factor
                    factor = Factor(
                        name=factor_def["name"],
                        description=factor_def["description"],
                        formula=factor_def["formula"],
                        type=factor_def["type"],
                        status=factor_def["status"],
                        group_id=group.id
                    )
                    db.add(factor)
                    created_count += 1
            
            if created_count > 0:
                db.commit()
                print(f"Created {created_count} actual Qlib factors for {group.name}")
            else:
                print(f"All actual Qlib factors already exist for {group.name}")
        else:
            print(f"All factors already exist for {group.name}")
except Exception as e:
    print(f"Skipping factor creation due to error: {e}")
    print("This is expected if the factors table doesn't have the group_id column yet.")
    print("Please recreate the database or run a migration to add the group_id column.")

# Try to load real stock data from QLib
from app.services.qlib_service import qlib_service

try:
    # Initialize QLib
    qlib_service.init_qlib()
    
    # Get real instruments from QLib
    real_instruments = qlib_service.get_instruments()
    print(f"Got {len(real_instruments)} real instruments from QLib")
    
    # Use the first 4 instruments for sample data
    stock_codes = real_instruments[:4] if len(real_instruments) >= 4 else real_instruments
    
    # If no real instruments available, fall back to sample codes
    if not stock_codes:
        stock_codes = ["SH600000", "SH600001", "SZ000001", "SZ000002"]
    
    print(f"Using stock codes: {stock_codes}")
    
    # Get current date and past 30 days
    end_date = datetime.now()
    start_date = end_date - timedelta(days=30)
    
    # Format dates for QLib
    start_date_str = start_date.strftime("%Y-%m-%d")
    end_date_str = end_date.strftime("%Y-%m-%d")
    
    # Load data for each stock code
    for code in stock_codes:
        try:
            # Get real stock data from QLib
            stock_data_dict = qlib_service.get_stock_data(code, start_date_str, end_date_str)
            
            if stock_data_dict:
                # Insert real data into database
                for (date_str, field), value in stock_data_dict.items():
                    # Parse date
                    stock_date = datetime.strptime(date_str[0], "%Y-%m-%d").date()
                    
                    # Check if data already exists
                    existing_data = db.query(StockData).filter(
                        StockData.stock_code == code,
                        StockData.date == stock_date
                    ).first()
                    
                    if not existing_data:
                        # Create new stock data entry
                        stock_data = StockData(
                            stock_code=code,
                            date=stock_date,
                            open=stock_data_dict.get((date_str[0], "open"), 0),
                            high=stock_data_dict.get((date_str[0], "high"), 0),
                            low=stock_data_dict.get((date_str[0], "low"), 0),
                            close=stock_data_dict.get((date_str[0], "close"), 0),
                            volume=stock_data_dict.get((date_str[0], "volume"), 0)
                        )
                        db.add(stock_data)
                
                db.commit()
                print(f"Real data for stock {code} loaded from QLib and saved to database")
            else:
                # Fall back to generating sample data if QLib data not available
                print(f"No real data available for {code}, generating sample data")
                dates = [datetime.now() - timedelta(days=i) for i in range(30)]
                for date in dates:
                    existing_data = db.query(StockData).filter(
                        StockData.stock_code == code,
                        StockData.date == date.date()
                    ).first()
                    if not existing_data:
                        # Generate realistic stock data based on stock code and date
                        base_price = 10.0
                        
                        # Generate realistic price movements
                        open_price = base_price + (date.day * 0.01)
                        high_price = open_price + 0.5
                        low_price = open_price - 0.5
                        close_price = open_price + (date.day % 3 - 1) * 0.1
                        volume = 1000000 + (date.day * 10000) + (hash(code) % 100000)
                        
                        stock_data = StockData(
                            stock_code=code,
                            date=date.date(),
                            open=open_price,
                            high=high_price,
                            low=low_price,
                            close=close_price,
                            volume=volume
                        )
                        db.add(stock_data)
                db.commit()
                print(f"Sample data for stock {code} created successfully")
        except Exception as e:
            print(f"Error loading data for {code}: {e}")
            continue
except Exception as e:
    print(f"Error loading data from QLib: {e}")
    print("Falling back to sample data generation")
    
    # Fall back to sample data generation
    stock_codes = ["SH600000", "SH600001", "SZ000001", "SZ000002"]
    dates = [datetime.now() - timedelta(days=i) for i in range(30)]
    
    for code in stock_codes:
        for date in dates:
            existing_data = db.query(StockData).filter(
                StockData.stock_code == code,
                StockData.date == date.date()
            ).first()
            if not existing_data:
                # Generate realistic stock data based on stock code and date
                base_price = {
                    "SH600000": 8.0,
                    "SH600001": 4.0,
                    "SZ000001": 15.0,
                    "SZ000002": 20.0
                }.get(code, 10.0)
                
                # Generate realistic price movements
                open_price = base_price + (date.day * 0.01)
                high_price = open_price + 0.5
                low_price = open_price - 0.5
                close_price = open_price + (date.day % 3 - 1) * 0.1
                volume = 1000000 + (date.day * 10000) + (hash(code) % 100000)
                
                stock_data = StockData(
                    stock_code=code,
                    date=date.date(),
                    open=open_price,
                    high=high_price,
                    low=low_price,
                    close=close_price,
                    volume=volume
                )
                db.add(stock_data)
        db.commit()
        print(f"Sample data for stock {code} created successfully")

db.commit()
db.close()
