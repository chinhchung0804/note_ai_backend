"""
CASSO payment integration for Vietnamese users
CASSO provides automatic bank transfer detection with QR code
Easier and faster for users - no redirect needed
"""
import os
import hmac
import hashlib
import requests
from typing import Dict, Optional, List
from datetime import datetime, timedelta
import uuid


class CassoConfig:
    """CASSO configuration"""
    API_KEY = os.getenv("CASSO_API_KEY", "your_casso_api_key")
    BASE_URL = os.getenv("CASSO_BASE_URL", "https://oauth.casso.vn/v2")
    WEBHOOK_SECRET = os.getenv("CASSO_WEBHOOK_SECRET", "your_webhook_secret")
    
    # Bank account info for QR code
    BANK_ACCOUNT_ID = os.getenv("CASSO_BANK_ACCOUNT_ID", "")  # ID tài khoản ngân hàng trong CASSO
    BANK_NAME = os.getenv("BANK_NAME", "Vietcombank")
    BANK_ACCOUNT_NUMBER = os.getenv("BANK_ACCOUNT_NUMBER", "1234567890")
    BANK_ACCOUNT_NAME = os.getenv("BANK_ACCOUNT_NAME", "CONG TY NOTALLYX")


class CassoService:
    """CASSO payment service"""
    
    def __init__(self):
        self.api_key = CassoConfig.API_KEY
        self.base_url = CassoConfig.BASE_URL
        self.headers = {
            "Authorization": f"Apikey {self.api_key}",
            "Content-Type": "application/json"
        }
    
    def create_payment_request(
        self,
        order_id: str,
        amount: int,
        description: str,
        user_email: str = None
    ) -> Dict:
        """
        Create payment request with QR code
        
        Args:
            order_id: Unique order ID (will be used as transfer content)
            amount: Amount in VND
            description: Payment description
            user_email: User email for notification (optional)
        
        Returns:
            Dict with payment info including QR code URL
        """
        # Tạo nội dung chuyển khoản (transfer content)
        # Format: NOTALLYX <order_id>
        transfer_content = f"NOTALLYX {order_id}"
        
        # Tạo QR code URL (VietQR standard)
        qr_data = self._generate_vietqr_url(
            bank_account=CassoConfig.BANK_ACCOUNT_NUMBER,
            amount=amount,
            description=transfer_content
        )
        
        payment_info = {
            "order_id": order_id,
            "amount": amount,
            "currency": "VND",
            "transfer_content": transfer_content,
            "bank_info": {
                "bank_name": CassoConfig.BANK_NAME,
                "account_number": CassoConfig.BANK_ACCOUNT_NUMBER,
                "account_name": CassoConfig.BANK_ACCOUNT_NAME
            },
            "qr_code_url": qr_data["qr_url"],
            "qr_data_url": qr_data["qr_data_url"],
            "description": description,
            "status": "pending",
            "created_at": datetime.now().isoformat(),
            "expires_at": (datetime.now() + timedelta(minutes=15)).isoformat(),
            "instructions": [
                f"1. Quét mã QR bằng app ngân hàng",
                f"2. Hoặc chuyển khoản thủ công:",
                f"   - Ngân hàng: {CassoConfig.BANK_NAME}",
                f"   - Số TK: {CassoConfig.BANK_ACCOUNT_NUMBER}",
                f"   - Chủ TK: {CassoConfig.BANK_ACCOUNT_NAME}",
                f"   - Số tiền: {amount:,} VND",
                f"   - Nội dung: {transfer_content}",
                f"3. Hệ thống tự động xác nhận sau 1-2 phút"
            ]
        }
        
        return payment_info
    
    def _generate_vietqr_url(
        self,
        bank_account: str,
        amount: int,
        description: str
    ) -> Dict:
        """
        Generate VietQR URL for QR code
        
        VietQR is the standard QR code format for Vietnamese banks
        Supported by all major banks in Vietnam
        """
        # VietQR format
        # https://img.vietqr.io/image/{BANK_CODE}-{ACCOUNT_NUMBER}-{TEMPLATE}.png?amount={AMOUNT}&addInfo={DESCRIPTION}
        
        # Lấy bank code từ bank name
        bank_code = self._get_bank_code(CassoConfig.BANK_NAME)
        
        # Encode description for URL
        import urllib.parse
        encoded_desc = urllib.parse.quote(description)
        
        # Template: compact (nhỏ gọn), compact2 (có logo), print (in ấn)
        template = "compact2"
        
        qr_url = (
            f"https://img.vietqr.io/image/{bank_code}-{bank_account}-{template}.png"
            f"?amount={amount}&addInfo={encoded_desc}&accountName={urllib.parse.quote(CassoConfig.BANK_ACCOUNT_NAME)}"
        )
        
        # QR data URL for embedding
        qr_data_url = (
            f"https://api.vietqr.io/v2/generate"
        )
        
        return {
            "qr_url": qr_url,
            "qr_data_url": qr_data_url,
            "bank_code": bank_code
        }
    
    def _get_bank_code(self, bank_name: str) -> str:
        """Get bank code from bank name"""
        bank_codes = {
            "Vietcombank": "VCB",
            "VCB": "VCB",
            "Techcombank": "TCB",
            "TCB": "TCB",
            "BIDV": "BIDV",
            "VietinBank": "CTG",
            "CTG": "CTG",
            "Agribank": "AGR",
            "AGR": "AGR",
            "MB": "MB",
            "MBBank": "MB",
            "ACB": "ACB",
            "VPBank": "VPB",
            "VPB": "VPB",
            "TPBank": "TPB",
            "TPB": "TPB",
            "Sacombank": "STB",
            "STB": "STB",
            "HDBank": "HDB",
            "HDB": "HDB",
            "VIB": "VIB",
            "SHB": "SHB",
            "Eximbank": "EIB",
            "EIB": "EIB",
            "MSB": "MSB",
            "OCB": "OCB",
            "SeABank": "SEAB",
            "SEAB": "SEAB"
        }
        return bank_codes.get(bank_name, "VCB")
    
    def get_transactions(
        self,
        from_date: Optional[str] = None,
        to_date: Optional[str] = None,
        page: int = 1,
        page_size: int = 20
    ) -> List[Dict]:
        """
        Get transactions from CASSO
        
        Args:
            from_date: Start date (YYYY-MM-DD)
            to_date: End date (YYYY-MM-DD)
            page: Page number
            page_size: Items per page
        
        Returns:
            List of transactions
        """
        try:
            url = f"{self.base_url}/transactions"
            params = {
                "page": page,
                "pageSize": page_size
            }
            
            if from_date:
                params["fromDate"] = from_date
            if to_date:
                params["toDate"] = to_date
            
            response = requests.get(url, headers=self.headers, params=params)
            response.raise_for_status()
            
            data = response.json()
            return data.get("data", {}).get("records", [])
        
        except Exception as e:
            print(f"Error getting transactions from CASSO: {e}")
            return []
    
    def verify_transaction(
        self,
        order_id: str,
        expected_amount: int,
        time_window_minutes: int = 30
    ) -> Optional[Dict]:
        """
        Verify if a transaction exists for the order
        
        Args:
            order_id: Order ID to search for
            expected_amount: Expected amount in VND
            time_window_minutes: Time window to search (default 30 minutes)
        
        Returns:
            Transaction data if found, None otherwise
        """
        # Tìm kiếm trong transactions gần đây
        from_date = (datetime.now() - timedelta(minutes=time_window_minutes)).strftime("%Y-%m-%d")
        transactions = self.get_transactions(from_date=from_date)
        
        # Tìm transaction khớp với order_id và amount
        transfer_content = f"NOTALLYX {order_id}"
        
        for txn in transactions:
            # Kiểm tra nội dung chuyển khoản
            description = txn.get("description", "").upper()
            amount = int(txn.get("amount", 0))
            
            # Kiểm tra khớp
            if transfer_content.upper() in description and amount == expected_amount:
                return {
                    "transaction_id": txn.get("id"),
                    "amount": amount,
                    "description": txn.get("description"),
                    "when": txn.get("when"),
                    "bank_sub_acc_id": txn.get("bank_sub_acc_id"),
                    "verified": True
                }
        
        return None
    
    def verify_webhook_signature(
        self,
        payload: str,
        signature: str
    ) -> bool:
        """
        Verify webhook signature from CASSO
        
        Args:
            payload: Webhook payload (raw string)
            signature: Signature from header
        
        Returns:
            True if valid, False otherwise
        """
        try:
            # CASSO uses HMAC-SHA256
            expected_signature = hmac.new(
                CassoConfig.WEBHOOK_SECRET.encode('utf-8'),
                payload.encode('utf-8'),
                hashlib.sha256
            ).hexdigest()
            
            return hmac.compare_digest(expected_signature, signature)
        
        except Exception as e:
            print(f"Error verifying webhook signature: {e}")
            return False
    
    def process_webhook(self, webhook_data: Dict) -> Dict:
        """
        Process webhook from CASSO
        
        Args:
            webhook_data: Webhook data from CASSO
        
        Returns:
            Processed transaction info
        """
        # Extract transaction info
        transaction = webhook_data.get("data", {})
        
        # Parse order_id from description
        description = transaction.get("description", "")
        order_id = None
        
        # Format: "NOTALLYX <order_id>"
        if "NOTALLYX" in description.upper():
            parts = description.upper().split("NOTALLYX")
            if len(parts) > 1:
                order_id = parts[1].strip()
        
        return {
            "order_id": order_id,
            "transaction_id": transaction.get("id"),
            "amount": int(transaction.get("amount", 0)),
            "description": description,
            "when": transaction.get("when"),
            "bank_sub_acc_id": transaction.get("bank_sub_acc_id"),
            "verified": True
        }


# Pricing plans (same as before)
PRICING_PLANS = {
    "pro_1_month": {
        "id": "pro_1_month",
        "name": "Pro - 1 Tháng",
        "price_vnd": 99000,
        "price_usd": 4.99,
        "months": 1,
        "description": "Ghi chú không giới hạn, AI tiên tiến, hỗ trợ ưu tiên"
    },
    "pro_3_months": {
        "id": "pro_3_months",
        "name": "Pro - 3 Tháng",
        "price_vnd": 249000,
        "price_usd": 12.99,
        "months": 3,
        "description": "Tiết kiệm 16% - Ghi chú không giới hạn, AI tiên tiến, hỗ trợ ưu tiên"
    },
    "pro_12_months": {
        "id": "pro_12_months",
        "name": "Pro - 12 Tháng",
        "price_vnd": 990000,
        "price_usd": 49.99,
        "months": 12,
        "description": "Tiết kiệm 17% - Ghi chú không giới hạn, AI tiên tiến, hỗ trợ ưu tiên"
    }
}


def get_pricing_plan(plan_id: str) -> Optional[Dict]:
    """Get pricing plan details"""
    return PRICING_PLANS.get(plan_id)


def get_all_pricing_plans() -> List[Dict]:
    """Get all pricing plans"""
    return list(PRICING_PLANS.values())
