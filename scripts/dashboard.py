#!/usr/bin/env python3
"""Trading dashboard for Footbe-Trader."""

import json
import sqlite3
from datetime import datetime, timedelta
from pathlib import Path

from flask import Flask, render_template_string, jsonify

app = Flask(__name__)

# Database path
DB_PATH = Path(__file__).parent.parent / "data" / "footbe_dev.db"

DASHBOARD_HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Footbe Trader Dashboard</title>
    <script src="https://cdn.tailwindcss.com"></script>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        .card { @apply bg-white rounded-lg shadow-md p-6; }
        .stat-value { @apply text-3xl font-bold; }
        .stat-label { @apply text-gray-500 text-sm uppercase tracking-wide; }
        .positive { @apply text-green-600; }
        .negative { @apply text-red-600; }
        .neutral { @apply text-gray-600; }
    </style>
</head>
<body class="bg-gray-100 min-h-screen">
    <nav class="bg-indigo-600 text-white p-4 shadow-lg">
        <div class="container mx-auto flex justify-between items-center">
            <h1 class="text-2xl font-bold">⚽🏀 Footbe Trader</h1>
            <div class="text-sm">
                <span id="last-update">Last update: --</span>
                <button onclick="refreshData()" class="ml-4 bg-indigo-500 hover:bg-indigo-400 px-3 py-1 rounded">
                    Refresh
                </button>
            </div>
        </div>
    </nav>

    <main class="container mx-auto p-6">
        <!-- Summary Stats -->
        <div class="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-5 gap-4 mb-6">
            <div class="card">
                <div class="stat-label">Total P&L</div>
                <div class="stat-value" id="total-pnl">$0.00</div>
            </div>
            <div class="card">
                <div class="stat-label">Realized P&L</div>
                <div class="stat-value" id="realized-pnl">$0.00</div>
            </div>
            <div class="card">
                <div class="stat-label">Unrealized P&L</div>
                <div class="stat-value" id="unrealized-pnl">$0.00</div>
            </div>
            <div class="card">
                <div class="stat-label">Open Positions</div>
                <div class="stat-value neutral" id="position-count">0</div>
            </div>
            <div class="card">
                <div class="stat-label">Total Exposure</div>
                <div class="stat-value neutral" id="exposure">$0.00</div>
            </div>
        </div>

        <!-- Agent Status & Run Stats -->
        <div class="grid grid-cols-1 lg:grid-cols-3 gap-4 mb-6">
            <div class="card">
                <h2 class="text-lg font-semibold mb-4">Agent Status</h2>
                <div class="space-y-2">
                    <div class="flex justify-between">
                        <span class="text-gray-500">Status</span>
                        <span id="agent-status" class="font-medium">--</span>
                    </div>
                    <div class="flex justify-between">
                        <span class="text-gray-500">Last Run</span>
                        <span id="last-run" class="font-medium">--</span>
                    </div>
                    <div class="flex justify-between">
                        <span class="text-gray-500">Total Runs</span>
                        <span id="total-runs" class="font-medium">--</span>
                    </div>
                    <div class="flex justify-between">
                        <span class="text-gray-500">Run Type</span>
                        <span id="run-type" class="font-medium">--</span>
                    </div>
                </div>
            </div>
            <div class="card">
                <h2 class="text-lg font-semibold mb-4">Today's Activity</h2>
                <div class="space-y-2">
                    <div class="flex justify-between">
                        <span class="text-gray-500">Fixtures Evaluated</span>
                        <span id="fixtures-today" class="font-medium">--</span>
                    </div>
                    <div class="flex justify-between">
                        <span class="text-gray-500">Markets Evaluated</span>
                        <span id="markets-today" class="font-medium">--</span>
                    </div>
                    <div class="flex justify-between">
                        <span class="text-gray-500">Decisions Made</span>
                        <span id="decisions-today" class="font-medium">--</span>
                    </div>
                    <div class="flex justify-between">
                        <span class="text-gray-500">Orders Placed</span>
                        <span id="orders-today" class="font-medium">--</span>
                    </div>
                </div>
            </div>
            <div class="card">
                <h2 class="text-lg font-semibold mb-4">Order Stats</h2>
                <div class="space-y-2">
                    <div class="flex justify-between">
                        <span class="text-gray-500">Orders Filled</span>
                        <span id="orders-filled" class="font-medium text-green-600">--</span>
                    </div>
                    <div class="flex justify-between">
                        <span class="text-gray-500">Orders Pending</span>
                        <span id="orders-pending" class="font-medium text-yellow-600">--</span>
                    </div>
                    <div class="flex justify-between">
                        <span class="text-gray-500">Orders Rejected</span>
                        <span id="orders-rejected" class="font-medium text-red-600">--</span>
                    </div>
                    <div class="flex justify-between">
                        <span class="text-gray-500">Fill Rate</span>
                        <span id="fill-rate" class="font-medium">--</span>
                    </div>
                </div>
            </div>
        </div>

        <!-- Sport Breakdown -->
        <div class="grid grid-cols-1 lg:grid-cols-2 gap-4 mb-6">
            <div class="card">
                <h2 class="text-lg font-semibold mb-4">⚽ Soccer Positions</h2>
                <div class="overflow-x-auto">
                    <table class="w-full text-sm">
                        <thead>
                            <tr class="border-b">
                                <th class="text-left py-2">Market</th>
                                <th class="text-right py-2">Qty</th>
                                <th class="text-right py-2">Avg Price</th>
                                <th class="text-right py-2">Mark</th>
                                <th class="text-right py-2">P&L</th>
                            </tr>
                        </thead>
                        <tbody id="soccer-positions">
                        </tbody>
                    </table>
                </div>
            </div>
            <div class="card">
                <h2 class="text-lg font-semibold mb-4">🏀 NBA Positions</h2>
                <div class="overflow-x-auto">
                    <table class="w-full text-sm">
                        <thead>
                            <tr class="border-b">
                                <th class="text-left py-2">Market</th>
                                <th class="text-right py-2">Qty</th>
                                <th class="text-right py-2">Avg Price</th>
                                <th class="text-right py-2">Mark</th>
                                <th class="text-right py-2">P&L</th>
                            </tr>
                        </thead>
                        <tbody id="nba-positions">
                        </tbody>
                    </table>
                </div>
            </div>
        </div>

        <!-- P&L Chart -->
        <div class="card mb-6">
            <h2 class="text-lg font-semibold mb-4">P&L Over Time</h2>
            <canvas id="pnl-chart" height="100"></canvas>
        </div>

        <!-- Recent Decisions -->
        <div class="card mb-6">
            <h2 class="text-lg font-semibold mb-4">Recent Decisions</h2>
            <div class="overflow-x-auto">
                <table class="w-full text-sm">
                    <thead>
                        <tr class="border-b">
                            <th class="text-left py-2">Time</th>
                            <th class="text-left py-2">Market</th>
                            <th class="text-left py-2">Outcome</th>
                            <th class="text-left py-2">Action</th>
                            <th class="text-right py-2">Edge</th>
                            <th class="text-right py-2">Price</th>
                            <th class="text-left py-2">Status</th>
                        </tr>
                    </thead>
                    <tbody id="recent-decisions">
                    </tbody>
                </table>
            </div>
        </div>

        <!-- Recent Runs -->
        <div class="card">
            <h2 class="text-lg font-semibold mb-4">Recent Runs</h2>
            <div class="overflow-x-auto">
                <table class="w-full text-sm">
                    <thead>
                        <tr class="border-b">
                            <th class="text-left py-2">Run #</th>
                            <th class="text-left py-2">Started</th>
                            <th class="text-left py-2">Status</th>
                            <th class="text-right py-2">Fixtures</th>
                            <th class="text-right py-2">Markets</th>
                            <th class="text-right py-2">Decisions</th>
                            <th class="text-right py-2">Orders</th>
                            <th class="text-right py-2">Filled</th>
                            <th class="text-right py-2">P&L</th>
                        </tr>
                    </thead>
                    <tbody id="recent-runs">
                    </tbody>
                </table>
            </div>
        </div>
    </main>

    <footer class="bg-gray-800 text-gray-400 p-4 mt-8">
        <div class="container mx-auto text-center text-sm">
            Footbe Trader Dashboard | Auto-refreshes every 60 seconds
        </div>
    </footer>

    <script>
        let pnlChart = null;

        async function fetchData() {
            const response = await fetch('/api/data');
            return response.json();
        }

        function formatPnl(value) {
            const formatted = '$' + Math.abs(value).toFixed(2);
            return value >= 0 ? formatted : '-' + formatted;
        }

        function getPnlClass(value) {
            if (value > 0) return 'positive';
            if (value < 0) return 'negative';
            return 'neutral';
        }

        function formatTime(timestamp) {
            if (!timestamp) return '--';
            const date = new Date(timestamp + 'Z');
            return date.toLocaleTimeString();
        }

        function formatDateTime(timestamp) {
            if (!timestamp) return '--';
            const date = new Date(timestamp + 'Z');
            return date.toLocaleString();
        }

        function truncateTicker(ticker) {
            // Shorten long tickers for display
            if (ticker.length > 30) {
                return ticker.substring(0, 27) + '...';
            }
            return ticker;
        }

        async function refreshData() {
            try {
                const data = await fetchData();
                
                // Update summary stats
                const totalPnl = data.summary.realized_pnl + data.summary.unrealized_pnl;
                document.getElementById('total-pnl').textContent = formatPnl(totalPnl);
                document.getElementById('total-pnl').className = 'stat-value ' + getPnlClass(totalPnl);
                
                document.getElementById('realized-pnl').textContent = formatPnl(data.summary.realized_pnl);
                document.getElementById('realized-pnl').className = 'stat-value ' + getPnlClass(data.summary.realized_pnl);
                
                document.getElementById('unrealized-pnl').textContent = formatPnl(data.summary.unrealized_pnl);
                document.getElementById('unrealized-pnl').className = 'stat-value ' + getPnlClass(data.summary.unrealized_pnl);
                
                document.getElementById('position-count').textContent = data.summary.position_count;
                document.getElementById('exposure').textContent = formatPnl(data.summary.exposure);
                
                // Agent status
                document.getElementById('agent-status').textContent = data.agent.status;
                document.getElementById('agent-status').className = 'font-medium ' + 
                    (data.agent.status === 'running' ? 'text-green-600' : 
                     data.agent.status === 'completed' ? 'text-blue-600' : 'text-red-600');
                document.getElementById('last-run').textContent = formatDateTime(data.agent.last_run);
                document.getElementById('total-runs').textContent = data.agent.total_runs;
                document.getElementById('run-type').textContent = data.agent.run_type;
                
                // Today's activity
                document.getElementById('fixtures-today').textContent = data.today.fixtures;
                document.getElementById('markets-today').textContent = data.today.markets;
                document.getElementById('decisions-today').textContent = data.today.decisions;
                document.getElementById('orders-today').textContent = data.today.orders;
                
                // Order stats
                document.getElementById('orders-filled').textContent = data.orders.filled;
                document.getElementById('orders-pending').textContent = data.orders.pending;
                document.getElementById('orders-rejected').textContent = data.orders.rejected;
                document.getElementById('fill-rate').textContent = data.orders.fill_rate + '%';
                
                // Soccer positions
                const soccerBody = document.getElementById('soccer-positions');
                soccerBody.innerHTML = data.positions.soccer.map(p => `
                    <tr class="border-b hover:bg-gray-50">
                        <td class="py-2" title="${p.ticker}">${truncateTicker(p.ticker)}</td>
                        <td class="text-right">${p.quantity}</td>
                        <td class="text-right">$${p.avg_price.toFixed(2)}</td>
                        <td class="text-right">$${p.mark_price.toFixed(2)}</td>
                        <td class="text-right ${getPnlClass(p.unrealized_pnl)}">${formatPnl(p.unrealized_pnl)}</td>
                    </tr>
                `).join('') || '<tr><td colspan="5" class="py-4 text-center text-gray-500">No positions</td></tr>';
                
                // NBA positions
                const nbaBody = document.getElementById('nba-positions');
                nbaBody.innerHTML = data.positions.nba.map(p => `
                    <tr class="border-b hover:bg-gray-50">
                        <td class="py-2" title="${p.ticker}">${truncateTicker(p.ticker)}</td>
                        <td class="text-right">${p.quantity}</td>
                        <td class="text-right">$${p.avg_price.toFixed(2)}</td>
                        <td class="text-right">$${p.mark_price.toFixed(2)}</td>
                        <td class="text-right ${getPnlClass(p.unrealized_pnl)}">${formatPnl(p.unrealized_pnl)}</td>
                    </tr>
                `).join('') || '<tr><td colspan="5" class="py-4 text-center text-gray-500">No positions</td></tr>';
                
                // Recent decisions
                const decisionsBody = document.getElementById('recent-decisions');
                decisionsBody.innerHTML = data.decisions.map(d => {
                    const actionClass = d.action === 'buy' ? 'text-green-600' : 
                                       d.action === 'sell' ? 'text-red-600' : 'text-gray-500';
                    const statusClass = d.order_placed ? 'text-green-600' : 'text-gray-500';
                    return `
                        <tr class="border-b hover:bg-gray-50">
                            <td class="py-2">${formatTime(d.timestamp)}</td>
                            <td title="${d.ticker}">${truncateTicker(d.ticker)}</td>
                            <td>${d.outcome}</td>
                            <td class="${actionClass} font-medium">${d.action.toUpperCase()}</td>
                            <td class="text-right">${d.edge ? (d.edge * 100).toFixed(1) + '%' : '--'}</td>
                            <td class="text-right">${d.price ? '$' + d.price.toFixed(2) : '--'}</td>
                            <td class="${statusClass}">${d.order_placed ? '✓ Placed' : 'Skipped'}</td>
                        </tr>
                    `;
                }).join('') || '<tr><td colspan="7" class="py-4 text-center text-gray-500">No decisions yet</td></tr>';
                
                // Recent runs
                const runsBody = document.getElementById('recent-runs');
                runsBody.innerHTML = data.runs.map(r => {
                    const pnl = r.realized_pnl + r.unrealized_pnl;
                    const statusClass = r.status === 'completed' ? 'text-green-600' : 
                                        r.status === 'running' ? 'text-blue-600' : 'text-red-600';
                    return `
                        <tr class="border-b hover:bg-gray-50">
                            <td class="py-2 font-medium">#${r.id}</td>
                            <td>${formatDateTime(r.started_at)}</td>
                            <td class="${statusClass}">${r.status}</td>
                            <td class="text-right">${r.fixtures}</td>
                            <td class="text-right">${r.markets}</td>
                            <td class="text-right">${r.decisions}</td>
                            <td class="text-right">${r.orders}</td>
                            <td class="text-right">${r.filled}</td>
                            <td class="text-right ${getPnlClass(pnl)}">${formatPnl(pnl)}</td>
                        </tr>
                    `;
                }).join('');
                
                // P&L Chart
                updatePnlChart(data.pnl_history);
                
                // Update timestamp
                document.getElementById('last-update').textContent = 'Last update: ' + new Date().toLocaleTimeString();
                
            } catch (error) {
                console.error('Error fetching data:', error);
            }
        }

        function updatePnlChart(history) {
            const ctx = document.getElementById('pnl-chart').getContext('2d');
            
            if (pnlChart) {
                pnlChart.destroy();
            }
            
            pnlChart = new Chart(ctx, {
                type: 'line',
                data: {
                    labels: history.map(h => formatDateTime(h.timestamp)),
                    datasets: [
                        {
                            label: 'Total P&L',
                            data: history.map(h => h.realized + h.unrealized),
                            borderColor: 'rgb(99, 102, 241)',
                            backgroundColor: 'rgba(99, 102, 241, 0.1)',
                            fill: true,
                            tension: 0.3
                        },
                        {
                            label: 'Realized P&L',
                            data: history.map(h => h.realized),
                            borderColor: 'rgb(34, 197, 94)',
                            backgroundColor: 'transparent',
                            borderDash: [5, 5],
                            tension: 0.3
                        }
                    ]
                },
                options: {
                    responsive: true,
                    plugins: {
                        legend: {
                            position: 'top',
                        }
                    },
                    scales: {
                        y: {
                            beginAtZero: false,
                            ticks: {
                                callback: function(value) {
                                    return '$' + value.toFixed(2);
                                }
                            }
                        }
                    }
                }
            });
        }

        // Initial load
        refreshData();
        
        // Auto-refresh every 60 seconds
        setInterval(refreshData, 60000);
    </script>
</body>
</html>
"""


def get_db():
    """Get database connection."""
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    return conn


@app.route('/')
def index():
    """Serve the dashboard."""
    return render_template_string(DASHBOARD_HTML)


@app.route('/api/data')
def api_data():
    """API endpoint for dashboard data."""
    conn = get_db()
    cursor = conn.cursor()
    
    # Summary stats from latest run
    cursor.execute("""
        SELECT 
            total_realized_pnl, total_unrealized_pnl, 
            total_exposure, position_count,
            status, run_type, started_at
        FROM agent_runs 
        ORDER BY id DESC LIMIT 1
    """)
    latest_run = cursor.fetchone()
    
    summary = {
        "realized_pnl": latest_run["total_realized_pnl"] if latest_run else 0,
        "unrealized_pnl": latest_run["total_unrealized_pnl"] if latest_run else 0,
        "exposure": latest_run["total_exposure"] if latest_run else 0,
        "position_count": latest_run["position_count"] if latest_run else 0,
    }
    
    # Agent status
    cursor.execute("SELECT COUNT(*) as cnt FROM agent_runs")
    total_runs = cursor.fetchone()["cnt"]
    
    agent = {
        "status": latest_run["status"] if latest_run else "unknown",
        "run_type": latest_run["run_type"] if latest_run else "unknown",
        "last_run": latest_run["started_at"] if latest_run else None,
        "total_runs": total_runs,
    }
    
    # Today's activity (aggregate from all runs today)
    today = datetime.utcnow().strftime("%Y-%m-%d")
    cursor.execute("""
        SELECT 
            COALESCE(SUM(fixtures_evaluated), 0) as fixtures,
            COALESCE(SUM(markets_evaluated), 0) as markets,
            COALESCE(SUM(decisions_made), 0) as decisions,
            COALESCE(SUM(orders_placed), 0) as orders
        FROM agent_runs 
        WHERE date(started_at) = ?
    """, (today,))
    today_row = cursor.fetchone()
    
    today_stats = {
        "fixtures": today_row["fixtures"],
        "markets": today_row["markets"],
        "decisions": today_row["decisions"],
        "orders": today_row["orders"],
    }
    
    # Order stats
    cursor.execute("""
        SELECT 
            COALESCE(SUM(CASE WHEN status = 'filled' THEN 1 ELSE 0 END), 0) as filled,
            COALESCE(SUM(CASE WHEN status = 'pending' OR status = 'resting' THEN 1 ELSE 0 END), 0) as pending,
            COALESCE(SUM(CASE WHEN status = 'rejected' OR status = 'cancelled' THEN 1 ELSE 0 END), 0) as rejected,
            COUNT(*) as total
        FROM paper_orders
    """)
    order_row = cursor.fetchone()
    
    fill_rate = 0
    if order_row["total"] > 0:
        fill_rate = round(order_row["filled"] / order_row["total"] * 100, 1)
    
    orders = {
        "filled": order_row["filled"],
        "pending": order_row["pending"],
        "rejected": order_row["rejected"],
        "fill_rate": fill_rate,
    }
    
    # Positions split by sport
    cursor.execute("""
        SELECT ticker, quantity, average_entry_price, mark_price, unrealized_pnl
        FROM paper_positions
        WHERE quantity > 0
        ORDER BY ticker
    """)
    positions = cursor.fetchall()
    
    soccer_positions = []
    nba_positions = []
    
    for p in positions:
        pos_data = {
            "ticker": p["ticker"],
            "quantity": p["quantity"],
            "avg_price": p["average_entry_price"],
            "mark_price": p["mark_price"] or 0,
            "unrealized_pnl": p["unrealized_pnl"],
        }
        if "KXNBAGAME" in p["ticker"]:
            nba_positions.append(pos_data)
        else:
            soccer_positions.append(pos_data)
    
    # Recent decisions
    cursor.execute("""
        SELECT 
            timestamp, market_ticker, outcome, action, 
            order_placed, edge_calculation_json, order_params_json
        FROM decision_records
        ORDER BY timestamp DESC
        LIMIT 20
    """)
    decisions = []
    for row in cursor.fetchall():
        edge = None
        price = None
        if row["edge_calculation_json"]:
            try:
                edge_data = json.loads(row["edge_calculation_json"])
                edge = edge_data.get("edge")
            except:
                pass
        if row["order_params_json"]:
            try:
                order_data = json.loads(row["order_params_json"])
                price = order_data.get("price")
            except:
                pass
        
        decisions.append({
            "timestamp": row["timestamp"],
            "ticker": row["market_ticker"],
            "outcome": row["outcome"],
            "action": row["action"],
            "order_placed": bool(row["order_placed"]),
            "edge": edge,
            "price": price,
        })
    
    # Recent runs
    cursor.execute("""
        SELECT 
            id, started_at, status, 
            fixtures_evaluated, markets_evaluated, decisions_made,
            orders_placed, orders_filled,
            total_realized_pnl, total_unrealized_pnl
        FROM agent_runs
        ORDER BY id DESC
        LIMIT 10
    """)
    runs = []
    for row in cursor.fetchall():
        runs.append({
            "id": row["id"],
            "started_at": row["started_at"],
            "status": row["status"],
            "fixtures": row["fixtures_evaluated"],
            "markets": row["markets_evaluated"],
            "decisions": row["decisions_made"],
            "orders": row["orders_placed"],
            "filled": row["orders_filled"],
            "realized_pnl": row["total_realized_pnl"],
            "unrealized_pnl": row["total_unrealized_pnl"],
        })
    
    # P&L history
    cursor.execute("""
        SELECT started_at, total_realized_pnl, total_unrealized_pnl
        FROM agent_runs
        WHERE status = 'completed'
        ORDER BY id ASC
        LIMIT 50
    """)
    pnl_history = []
    for row in cursor.fetchall():
        pnl_history.append({
            "timestamp": row["started_at"],
            "realized": row["total_realized_pnl"],
            "unrealized": row["total_unrealized_pnl"],
        })
    
    conn.close()
    
    return jsonify({
        "summary": summary,
        "agent": agent,
        "today": today_stats,
        "orders": orders,
        "positions": {
            "soccer": soccer_positions,
            "nba": nba_positions,
        },
        "decisions": decisions,
        "runs": runs,
        "pnl_history": pnl_history,
    })


if __name__ == '__main__':
    print(f"Starting dashboard at http://localhost:5050")
    print(f"Database: {DB_PATH}")
    app.run(host='0.0.0.0', port=5050, debug=True)
