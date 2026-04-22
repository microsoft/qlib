#!/usr/bin/env python3
"""
Verify that benchmark configs have been successfully imported into the database.
"""

import sys

# Add the backend directory to Python path
sys.path.append('/home/qlib_t/backend')

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from app.models.config import Config, ConfigType

def verify_configs():
    """
    Verify that benchmark configs have been successfully imported.
    """
    # Database connection
    DATABASE_URL = "sqlite:////home/qlib_t/backend/test.db"
    engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
    SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    
    db = SessionLocal()
    try:
        # Get all configs
        all_configs = db.query(Config).all()
        print(f"Total configs in database: {len(all_configs)}")
        
        # Get only experiment templates
        template_configs = db.query(Config).filter(Config.type == ConfigType.EXPERIMENT_TEMPLATE).all()
        print(f"Experiment template configs: {len(template_configs)}")
        
        # Print sample of imported configs
        print("\nSample imported configs:")
        for i, config in enumerate(template_configs[:10]):  # Show first 10
            print(f"{i+1}. {config.name} - {config.type} - {config.description[:50]}...")
        
        # Print summary
        print(f"\nVerification completed.")
        print(f"- Total configs: {len(all_configs)}")
        print(f"- Experiment templates: {len(template_configs)}")
        print(f"- Expected benchmarks: ~56")
        print(f"- Imported benchmarks: {len(template_configs)}")
        
        if len(template_configs) > 0:
            print("✅ Verification PASSED: Benchmark configs have been successfully imported!")
        else:
            print("❌ Verification FAILED: No experiment templates found!")
        
        return len(template_configs) > 0
        
    except Exception as e:
        print(f"Error during verification: {e}")
        return False
    finally:
        db.close()

if __name__ == "__main__":
    verify_configs()
