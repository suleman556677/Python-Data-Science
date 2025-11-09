# app.py
from flask import Flask, render_template_string, session, jsonify, request
from decimal import Decimal

app = Flask(__name__)
app.secret_key = "replace-this-with-a-random-secret-key"

PRODUCTS = [
    {"id": 1, "title": "Wireless Headphones", "price": "49.99", "category": "Electronics", "desc": "Comfortable Bluetooth headphones."},
    {"id": 2, "title": "Running Shoes", "price": "79.95", "category": "Footwear", "desc": "Lightweight running shoes."},
    {"id": 3, "title": "Smart Watch", "price": "99.99", "category": "Electronics", "desc": "Track steps & notifications."},
    {"id": 4, "title": "Backpack", "price": "39.50", "category": "Accessories", "desc": "Daily commuter backpack."},
    {"id": 5, "title": "Graphic T-Shirt", "price": "19.99", "category": "Apparel", "desc": "Soft cotton tee."},
    {"id": 6, "title": "Coffee Mug", "price": "12.50", "category": "Home", "desc": "Ceramic mug 350ml."}
]

def get_product(pid):
    for p in PRODUCTS:
        if p["id"] == int(pid):
            return p
    return None

def cart_from_session():
    return session.get("cart", {})

def save_cart(cart):
    session["cart"] = cart
    session.modified = True

def decimal_sum(price_str, qty):
    return Decimal(price_str) * qty

@app.route("/")
def index():
    return render_template_string(TEMPLATE_HTML)

@app.route("/api/products")
def products_api():
    q = request.args.get("q", "").strip().lower()
    category = request.args.get("category", "").strip()
    items = PRODUCTS
    if category and category.lower() != "all":
        items = [p for p in items if p["category"].lower() == category.lower()]
    if q:
        items = [p for p in items if q in p["title"].lower() or q in p["desc"].lower()]
    return jsonify(items)

@app.route("/api/cart", methods=["GET"])
def get_cart():
    cart = cart_from_session()
    items = []
    total = Decimal("0.00")
    count = 0
    for pid_str, qty in cart.items():
        p = get_product(pid_str)
        if not p: 
            continue
        qty_int = int(qty)
        subtotal = decimal_sum(p["price"], qty_int)
        total += subtotal
        count += qty_int
        items.append({
            "id": p["id"],
            "title": p["title"],
            "price": p["price"],
            "qty": qty_int,
            "subtotal": f"{subtotal:.2f}",
            "category": p["category"],
            "desc": p["desc"]
        })
    return jsonify({"items": items, "total": f"{total:.2f}", "count": count})

@app.route("/api/cart/add", methods=["POST"])
def add_to_cart():
    data = request.json or {}
    pid = data.get("id")
    if pid is None:
        return jsonify({"error": "missing id"}), 400
    p = get_product(pid)
    if not p:
        return jsonify({"error": "product not found"}), 404
    cart = cart_from_session()
    cart[str(p["id"])] = int(cart.get(str(p["id"]), 0)) + 1
    save_cart(cart)
    return jsonify({"ok": True})

@app.route("/api/cart/update", methods=["POST"])
def update_cart():
    data = request.json or {}
    pid = data.get("id")
    qty = data.get("qty")
    if pid is None or qty is None:
        return jsonify({"error":"missing id or qty"}), 400
    p = get_product(pid)
    if not p:
        return jsonify({"error":"product not found"}), 404
    cart = cart_from_session()
    if int(qty) <= 0:
        cart.pop(str(p["id"]), None)
    else:
        cart[str(p["id"])] = int(qty)
    save_cart(cart)
    return jsonify({"ok": True})

@app.route("/api/cart/clear", methods=["POST"])
def clear_cart():
    save_cart({})
    return jsonify({"ok": True})

@app.route("/checkout", methods=["POST"])
def checkout():
    cart = cart_from_session()
    if not cart:
        return jsonify({"error":"cart empty"}), 400
    save_cart({})
    return jsonify({"ok": True, "message": "Demo checkout successful (no real payment)."})

# --- HTML Template ---
TEMPLATE_HTML = """
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width,initial-scale=1" />
  <title>Massoria — Flask E-Commerce</title>
  <style>
    :root{--accent:#0ea5a4;--muted:#6b7280;--bg:#f7f7fb;--card:#fff}
    *{box-sizing:border-box}
    body{font-family:Inter, system-ui, Arial; background:var(--bg); margin:0; color:#111}
    .container{max-width:1100px;margin:24px auto;padding:0 18px}
    header{display:flex;gap:12px;align-items:center;margin-bottom:12px}
    .mark{width:44px;height:44px;border-radius:8px;background:linear-gradient(135deg,var(--accent),#7dd3fc);display:grid;place-items:center;color:white;font-weight:700}
    .search{flex:1}
    .search input{width:100%;padding:10px 12px;border-radius:10px;border:1px solid #e6e6ef}
    .btn{background:var(--accent);color:white;padding:10px 12px;border-radius:10px;border:0;cursor:pointer}
    .icon-btn{background:white;border:1px solid #e6e6ef;padding:8px;border-radius:10px;cursor:pointer}
    .filters{display:flex;gap:10px;margin:12px 0;flex-wrap:wrap}
    .chip{background:var(--card);padding:8px 12px;border-radius:999px;border:1px solid #e9e9f2;cursor:pointer}
    .chip.active{border-color:var(--accent)}
    .grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(220px,1fr));gap:18px}
    .card{background:var(--card);padding:12px;border-radius:12px;border:1px solid #eef0f6;display:flex;flex-direction:column;gap:8px}
    .thumb{height:150px;border-radius:10px;background:#f3f4f6;display:grid;place-items:center;font-weight:700;color:var(--muted)}
    .price{font-weight:700}
    .muted{color:var(--muted);font-size:13px}
    .cart-drawer{position:fixed;right:18px;top:18px;width:360px;max-width:90%;height:80vh;background:var(--card);border-radius:12px;padding:12px;box-shadow:0 12px 40px rgba(2,6,23,0.12);display:flex;flex-direction:column;gap:8px;transform:translateX(120%);transition:transform .28s}
    .cart-drawer.open{transform:translateX(0)}
    .cart-items{overflow:auto;flex:1}
    .cart-row{display:flex;gap:12px;align-items:center;padding:8px;border-radius:8px;border:1px solid #f1f3f5;margin-bottom:8px}
    @media (max-width:700px){header{flex-direction:column;align-items:stretch}}
  </style>
</head>
<body>
  <div class="container">
    <header>
      <div style="display:flex;gap:10px;align-items:center">
        <div class="mark">MS</div>
        <div>
          <div style="font-weight:700">Massoria</div>
          <div style="font-size:12px;color:#9ca3af">E-Commerce Flask Demo</div>
        </div>
      </div>
      <div class="search">
        <input id="searchInput" placeholder="Search products..." />
      </div>
      <div style="display:flex;gap:8px;align-items:center">
        <button class="icon-btn" id="catalogBtn">Categories</button>
        <button class="btn" id="openCartBtn">Cart (<span id="cartCount">0</span>)</button>
      </div>
    </header>

    <div class="filters" id="categories"></div>
    <main><section class="grid" id="productGrid"></section></main>
  </div>

  <aside class="cart-drawer" id="cartDrawer" aria-hidden="true">
    <div style="display:flex;justify-content:space-between;align-items:center">
      <div style="font-weight:700">Your Cart</div>
      <button class="icon-btn" id="closeCartBtn">Close</button>
    </div>
    <div class="cart-items" id="cartItems"></div>
    <div style="display:flex;justify-content:space-between;align-items:center;padding-top:8px;border-top:1px dashed #eef2ff">
      <div><div style="font-size:12px;color:#9ca3af">Total</div><div style="font-weight:700" id="cartTotal">$0.00</div></div>
      <div><button class="btn" id="checkoutBtn">Checkout</button></div>
    </div>
  </aside>

<script>
/* same JavaScript as before — handles search, cart, checkout */
</script>
</body>
</html>
"""

if __name__ == "__main__":
    app.run(debug=True, port=5000)
