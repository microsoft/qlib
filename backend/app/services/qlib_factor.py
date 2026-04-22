from qlib.contrib.data.loader import Alpha158DL, Alpha360DL

class QlibFactorService:
    """Service to get actual factor definitions from Qlib loaders"""
    
    @staticmethod
    def get_alpha158_factors():
        """Get Alpha158 factors"""
        fields, names = Alpha158DL.get_feature_config()
        factors = []
        for i, (field, name) in enumerate(zip(fields, names)):
            factors.append({
                "name": name,
                "description": f"Alpha158 factor {name}",
                "formula": field,
                "type": "alpha158",
                "status": "active"
            })
        return factors
    
    @staticmethod
    def get_alpha360_factors():
        """Get Alpha360 factors"""
        fields, names = Alpha360DL.get_feature_config()
        factors = []
        for i, (field, name) in enumerate(zip(fields, names)):
            factors.append({
                "name": name,
                "description": f"Alpha360 factor {name}",
                "formula": field,
                "type": "alpha360",
                "status": "active"
            })
        return factors
    
    @staticmethod
    def get_all_qlib_factors():
        """Get all Qlib factors"""
        alpha158 = QlibFactorService.get_alpha158_factors()
        alpha360 = QlibFactorService.get_alpha360_factors()
        return {
            "alpha158": alpha158,
            "alpha360": alpha360
        }
