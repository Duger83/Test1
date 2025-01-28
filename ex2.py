import unittest
from unittest.mock import MagicMock

class CreditCard:
    
    def getCardNumber(self):
        return '1234 5678 9101 1213'

    def getCardHolder(self):
        return 'Card Holder'

    def getExpiryDate(self):
        return '01/01/2030'

    def getCVV(self):
        return '123'

    def charge(self, amount: float):
        if amount <= 0:
            raise ValueError('Попытка списания отрицательной суммы!')
        else:
            if amount<=1000:
                return 'Успешный платеж'
            else:
                raise ValueError('Попытка чрезмерного списания!')

class PaymentForm:
    def __init__(self, credit_card: CreditCard):
        self.credit_card = credit_card

    def pay(self, amount: float):
        return self.credit_card.charge(amount)

card = CreditCard()
print("Карта:" + card.getCardNumber())
print(f"Владелец: {card.getCardHolder()}")
print(f"Срок действия: {card.getExpiryDate()}")
print(f"CVC: {card.getCVV()}")

amount = 100.00

payment_form = PaymentForm(card)
result = payment_form.pay(amount)
print(result)

try:
    card.charge(1500.00)
except Exception as e:
    print(e)

result = payment_form.pay(50.00)
print(result)

class TestPaymentForm(unittest.TestCase):
    def setUp(self):
        self.mock_credit_card = MagicMock(spec=CreditCard)
        self.payment_form = PaymentForm(self.mock_credit_card)

    def test_pay_success(self):
        amount = 100
        self.payment_form.pay(amount)
        self.mock_credit_card.charge.assert_called_once_with(amount)
        
    def test_pay_negative_amount(self):
        amount = -10
        self.payment_form.pay(amount)
        self.mock_credit_card.charge.assert_called_once_with(amount)
        
    def test_pay_bigger_amount(self):
        amount = 100000
        self.payment_form.pay(amount)
        self.mock_credit_card.charge.assert_called_once_with(amount)        

if __name__ == '__main__':
    unittest.main()