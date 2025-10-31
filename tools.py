import datetime
import pandas as pd
from llama_index.core.tools import FunctionTool


class Tools:
    """Colección de tools relacionadas con devoluciones."""

    def __init__(self, pedidos_df: pd.DataFrame):
        """
        pedidos_df: DataFrame con columnas:
        [customer_id, product, category, price, quantity, order_date, payment_method]
        """
        self.df = pedidos_df.copy()

        # Normalizamos los nombres de columnas y texto
        self.df.columns = [c.lower().strip() for c in self.df.columns]
        self.df["product"] = self.df["product"].astype(str).str.lower()
        self.df["category"] = self.df["category"].astype(str).str.lower()

    # === TOOL PRINCIPAL ===
    def evaluate_return_eligibility(self, customer_id: str, product_name: str) -> str:
        """
        Evalúa si un pedido es elegible para devolución.
        Busca el pedido por customer_id y product_name y aplica reglas.
        """
        product_name = product_name.lower().strip()

        # Filtramos el pedido correspondiente
        pedido = self.df[
            (self.df["customer_id"] == customer_id)
            & (self.df["product"] == product_name)
        ]

        if pedido.empty:
            return f"No se encontró ningún pedido de '{product_name}' para el cliente {customer_id}."

        pedido = pedido.iloc[0]

        # 1. Categoría no elegible
        categoria = pedido["category"]
        if categoria in ["higiene", "alimentos"]:
            return f"❌ El producto '{pedido['product']}' pertenece a la categoría '{categoria}', que no admite devoluciones."

        # 2. Días desde la compra
        try:
            order_date = pd.to_datetime(pedido["order_date"]).date()
        except Exception:
            return f"⚠️ No se pudo interpretar la fecha de compra del pedido. Verifica el formato en 'order_date'."

        days_since_order = (datetime.date.today() - order_date).days

        if days_since_order > 30:
            return f"❌ Han pasado {days_since_order} días desde la compra. Solo se admiten devoluciones dentro de 30 días."

        # 3. Método de pago especial
        payment_method = pedido["payment_method"].lower()
        if payment_method == "efectivo":
            return (
                f"⚠️ Pedido elegible, pero requiere revisión manual "
                f"por haber sido pagado en efectivo. ({days_since_order} días desde la compra)."
            )

        # 4. Elegible
        return (
            f"✅ El producto '{pedido['product']}' del cliente {customer_id} "
            f"es elegible para devolución. ({days_since_order} días desde la compra)."
        )

    # === ENVOLVER EN FUNCTIONTOOL ===
    def get_tools(self):
        """Devuelve las tools listas para conectar al agente."""
        return [
            FunctionTool.from_defaults(
                fn=self.evaluate_return_eligibility,
                name="evaluate_return_eligibility",
                description=(
                    "Evalúa si un pedido es elegible para devolución. "
                    "Usa el ID del cliente y el nombre del producto."
                ),
            ),
        ]
