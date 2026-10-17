package main

import (
	"encoding/json"
	"fmt"
	"io"
	"log"
	"net/http"
	"os"
	"strings"
	"sync"
	"time"
)

// TradingSignal represents an algorithmic order signal from the forecasting feed
type TradingSignal struct {
	Timestamp       string  `json:"timestamp"`
	AssetID         string  `json:"asset_id"`
	Action          string  `json:"action"` // DEC_BUY_DA_SELL_RT, INC_SELL_DA_BUY_RT, HOLD
	VolumeMW        float64 `json:"volume_mw"`
	ExpectedSpread  float64 `json:"expected_spread"`
	DAPriceForecast float64 `json:"da_price_forecast"`
	RTPriceForecast float64 `json:"rt_price_forecast"`
	SpikeRisk       float64 `json:"spike_risk"`
	Rationale       string  `json:"rationale"`
}

// FIXOrderMessage represents an execution order sent to the exchange/ISO gateway
type FIXOrderMessage struct {
	MsgType       string    `json:"msg_type"` // "35=D" (New Order Single)
	ClOrdID       string    `json:"cl_ord_id"`
	Symbol        string    `json:"symbol"`
	Side          string    `json:"side"` // "1" = Buy, "2" = Sell
	OrderQty      float64   `json:"order_qty_mw"`
	Price         float64   `json:"limit_price_usd"`
	TransactTime  time.Time `json:"transact_time"`
	RiskStatus    string    `json:"risk_status"`
	LatencyMicros int64     `json:"latency_micros"`
}

// RiskEngine validates pre-trade risk controls
type RiskEngine struct {
	MaxPositionMW  float64
	MaxSpikeRisk   float64
	DailyBudgetUSD float64
	mu             sync.Mutex
	AllocatedMW    float64
}

func NewRiskEngine(maxMW, maxSpikeRisk float64) *RiskEngine {
	return &RiskEngine{
		MaxPositionMW: maxMW,
		MaxSpikeRisk:  maxSpikeRisk,
	}
}

func (r *RiskEngine) Validate(sig TradingSignal) (bool, string) {
	r.mu.Lock()
	defer r.mu.Unlock()

	if sig.Action == "HOLD" || sig.VolumeMW <= 0 {
		return false, "ACTION_HOLD"
	}
	if sig.VolumeMW > r.MaxPositionMW {
		return false, fmt.Sprintf("EXCEEDS_MAX_POSITION: %.1f MW > %.1f MW", sig.VolumeMW, r.MaxPositionMW)
	}
	if sig.SpikeRisk > r.MaxSpikeRisk && sig.Action == "INC_SELL_DA_BUY_RT" {
		return false, fmt.Sprintf("EXCESSIVE_SPIKE_RISK: %.1f%% > %.1f%%", sig.SpikeRisk*100, r.MaxSpikeRisk*100)
	}
	r.AllocatedMW += sig.VolumeMW
	return true, "APPROVED"
}

// TradingFeedHandler consumes the Python forecasting API
type TradingFeedHandler struct {
	APIEndpoint string
	Client      *http.Client
	Risk        *RiskEngine
}

func NewTradingFeedHandler(endpoint string, risk *RiskEngine) *TradingFeedHandler {
	return &TradingFeedHandler{
		APIEndpoint: endpoint,
		Client:      &http.Client{Timeout: 5 * time.Second},
		Risk:        risk,
	}
}

func (h *TradingFeedHandler) FetchSignals() ([]TradingSignal, error) {
	url := fmt.Sprintf("%s/trading/signals?horizon_hours=24", h.APIEndpoint)
	resp, err := h.Client.Get(url)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("unexpected status: %s", resp.Status)
	}

	body, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, err
	}

	var signals []TradingSignal
	if err := json.Unmarshal(body, &signals); err != nil {
		return nil, err
	}
	return signals, nil
}

func (h *TradingFeedHandler) RouteOrder(sig TradingSignal, idx int) FIXOrderMessage {
	start := time.Now()
	approved, reason := h.Risk.Validate(sig)

	side := "1" // Buy
	limitPrice := sig.DAPriceForecast
	if sig.Action == "INC_SELL_DA_BUY_RT" {
		side = "2" // Sell
	}

	order := FIXOrderMessage{
		MsgType:       "35=D (NewOrderSingle)",
		ClOrdID:       fmt.Sprintf("ORD-%s-%03d", time.Now().Format("20060102150405"), idx),
		Symbol:        "HUB_NORTH_DA_RT",
		Side:          side,
		OrderQty:      sig.VolumeMW,
		Price:         limitPrice,
		TransactTime:  time.Now().UTC(),
		RiskStatus:    reason,
		LatencyMicros: time.Since(start).Microseconds(),
	}
	return order
}

func main() {
	fmt.Println(strings.Repeat("=", 75))
	fmt.Println("  ⚡ HIGH-PERFORMANCE ENERGY TRADING FEED HANDLER & ORDER ROUTER (Go)")
	fmt.Println(strings.Repeat("=", 75))

	apiURL := os.Getenv("FORECAST_API_URL")
	if apiURL == "" {
		apiURL = "http://127.0.0.1:8000"
	}

	risk := NewRiskEngine(120.0, 0.45)
	handler := NewTradingFeedHandler(apiURL, risk)

	fmt.Printf("[INFO] Connecting to Forecast Feed Gateway at: %s\n", apiURL)
	signals, err := handler.FetchSignals()

	if err != nil {
		fmt.Printf("[WARN] Live API unavailable (%v). Activating Simulated Feed Mode...\n", err)
		// Offline high-fidelity fallback demonstration
		signals = []TradingSignal{
			{
				Timestamp:       "2026-10-18 14:00:00",
				AssetID:         "VIRTUAL_HUB_NORTH",
				Action:          "DEC_BUY_DA_SELL_RT",
				VolumeMW:        85.5,
				ExpectedSpread:  14.20,
				DAPriceForecast: 42.50,
				RTPriceForecast: 56.70,
				SpikeRisk:       0.12,
				Rationale:       "Forecasted RT premium +$14.20/MWh (Evening solar ramp down)",
			},
			{
				Timestamp:       "2026-10-18 15:00:00",
				AssetID:         "VIRTUAL_HUB_NORTH",
				Action:          "DEC_BUY_DA_SELL_RT",
				VolumeMW:        95.0,
				ExpectedSpread:  18.75,
				DAPriceForecast: 48.00,
				RTPriceForecast: 66.75,
				SpikeRisk:       0.28,
				Rationale:       "Forecasted RT premium +$18.75/MWh (Thermal peaker dispatch trigger)",
			},
			{
				Timestamp:       "2026-10-18 16:00:00",
				AssetID:         "VIRTUAL_HUB_NORTH",
				Action:          "INC_SELL_DA_BUY_RT",
				VolumeMW:        60.0,
				ExpectedSpread:  -11.50,
				DAPriceForecast: 55.00,
				RTPriceForecast: 43.50,
				SpikeRisk:       0.04,
				Rationale:       "Forecasted DA premium +$11.50/MWh (Over-hedged DA capacity)",
			},
			{
				Timestamp:       "2026-10-18 17:00:00",
				AssetID:         "VIRTUAL_HUB_NORTH",
				Action:          "HOLD",
				VolumeMW:        0.0,
				ExpectedSpread:  2.10,
				DAPriceForecast: 45.00,
				RTPriceForecast: 47.10,
				SpikeRisk:       0.02,
				Rationale:       "Spread within transaction cost deadband",
			},
		}
	}

	fmt.Printf("[INFO] Received %d Time Series Forecast Signal Tickets from feed.\n", len(signals))
	fmt.Println(strings.Repeat("-", 75))
	fmt.Printf("%-20s | %-20s | %-10s | %-12s | %-10s\n", "TIMESTAMP", "ACTION", "VOLUME(MW)", "SPREAD($)", "STATUS")
	fmt.Println(strings.Repeat("-", 75))

	var wg sync.WaitGroup
	orders := make([]FIXOrderMessage, len(signals))

	// Concurrent order processing and risk checking
	for i, sig := range signals {
		wg.Add(1)
		go func(idx int, s TradingSignal) {
			defer wg.Done()
			orders[idx] = handler.RouteOrder(s, idx)
		}(i, sig)
	}
	wg.Wait()

	approvedCount := 0
	for i, ord := range orders {
		sig := signals[i]
		statusColor := "🟢"
		if ord.RiskStatus != "APPROVED" {
			statusColor = "⚪"
		} else {
			approvedCount++
		}

		fmt.Printf("%-20s | %-20s | %8.1f MW | $%10.2f | %s %s\n",
			sig.Timestamp, sig.Action, sig.VolumeMW, sig.ExpectedSpread, statusColor, ord.RiskStatus)
	}

	fmt.Println(strings.Repeat("=", 75))
	fmt.Printf("[METRICS] Total Orders Generated: %d | Approved by Pre-Trade Risk: %d\n", len(orders), approvedCount)
	fmt.Printf("[LATENCY] Ingestion & Risk Filtering completed concurrently across all signals.\n")
	fmt.Println(strings.Repeat("=", 75))
}
