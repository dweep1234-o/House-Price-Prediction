const SVG_NS = "http://www.w3.org/2000/svg";
const svg = document.getElementById("locality-map");
const tooltip = document.getElementById("tooltip");
const mapFrame = document.querySelector(".map-frame");

const emptyState = document.getElementById("empty-state");
const resultContent = document.getElementById("result-content");
const resultAddress = document.getElementById("result-address");
const priceValue = document.getElementById("price-value");
const featureChips = document.getElementById("feature-chips");
const calcTbody = document.getElementById("calc-tbody");
const calcTotal = document.getElementById("calc-total");

let selectedGroup = null;

const currency = (n) =>
  n.toLocaleString("en-US", { style: "currency", currency: "USD", maximumFractionDigits: 0 });

function el(tag, attrs = {}) {
  const node = document.createElementNS(SVG_NS, tag);
  for (const [k, v] of Object.entries(attrs)) node.setAttribute(k, v);
  return node;
}

// ---------- Draw the map ----------

async function drawMap() {
  const res = await fetch("/api/locality");
  const data = await res.json();

  svg.setAttribute("viewBox", `0 0 ${data.map_width} ${data.map_height}`);

  const rowCount = data.streets.length;
  const marginY = 90;
  const rowGap = rowCount > 1 ? (data.map_height - 2 * marginY) / (rowCount - 1) : 0;

  // Roads
  data.streets.forEach((name, i) => {
    const y = marginY + i * rowGap;
    svg.appendChild(el("line", { x1: 20, y1: y, x2: data.map_width - 20, y2: y, class: "road" }));
    svg.appendChild(el("line", { x1: 20, y1: y, x2: data.map_width - 20, y2: y, class: "road-dash" }));
    const label = el("text", { x: 30, y: y - 22, class: "street-label" });
    label.textContent = `${name} Street`;
    svg.appendChild(label);
  });

  // Houses
  data.houses.forEach((house) => drawHouse(house));
}

function drawHouse(house) {
  const g = el("g", {
    class: "house-group",
    tabindex: "0",
    role: "button",
    "aria-label": `${house.address}, view price`,
    "data-id": house.id,
  });
  g.appendChild(el("path", {
    class: "house-glow",
    d: houseGlowPath(house.x, house.y),
  }));
  g.appendChild(el("path", {
    class: "house-roof",
    d: `M ${house.x - 13} ${house.y - 2} L ${house.x} ${house.y - 15} L ${house.x + 13} ${house.y - 2} Z`,
  }));
  g.appendChild(el("rect", {
    class: "house-body",
    x: house.x - 10, y: house.y - 2, width: 20, height: 16, rx: 1.5,
  }));

  g.addEventListener("mouseenter", (e) => showTooltip(house, e));
  g.addEventListener("mousemove", (e) => positionTooltip(e));
  g.addEventListener("mouseleave", hideTooltip);
  g.addEventListener("click", () => selectHouse(g, house.id));
  g.addEventListener("keydown", (e) => {
    if (e.key === "Enter" || e.key === " ") {
      e.preventDefault();
      selectHouse(g, house.id);
    }
  });

  svg.appendChild(g);
}

function houseGlowPath(x, y) {
  return `M ${x - 15} ${y - 4} L ${x} ${y - 18} L ${x + 15} ${y - 4} L ${x + 12} ${y - 4}
          L ${x + 12} ${y + 15} L ${x - 12} ${y + 15} L ${x - 12} ${y - 4} Z`;
}

// ---------- Tooltip ----------

function showTooltip(house, event) {
  const f = house.features;
  tooltip.innerHTML = `<strong>${house.address}</strong><br>${f.bedrooms} bd &middot; ${f.bathrooms} ba &middot; ${f.area_sqft} sq ft`;
  tooltip.hidden = false;
  positionTooltip(event);
}

function positionTooltip(event) {
  const rect = mapFrame.getBoundingClientRect();
  tooltip.style.left = `${event.clientX - rect.left}px`;
  tooltip.style.top = `${event.clientY - rect.top}px`;
}

function hideTooltip() {
  tooltip.hidden = true;
}

// ---------- Selection + prediction ----------

async function selectHouse(group, houseId) {
  if (selectedGroup) selectedGroup.classList.remove("selected");
  group.classList.add("selected");
  selectedGroup = group;

  hideTooltip();
  emptyState.hidden = true;
  resultContent.hidden = true; // brief flash-clear while loading

  try {
    const res = await fetch(`/api/predict/${houseId}`);
    const data = await res.json();

    if (!res.ok) {
      resultAddress.textContent = "Something went wrong";
      priceValue.textContent = "—";
      resultContent.hidden = false;
      return;
    }

    renderResult(data);
  } catch (err) {
    resultAddress.textContent = "Could not reach the prediction service.";
    resultContent.hidden = false;
  }
}

function renderResult(data) {
  resultAddress.textContent = data.address;
  priceValue.textContent = currency(data.predicted_price);

  const f = data.features;
  featureChips.innerHTML = `
    <span class="feature-chip">${f.bedrooms} bd</span>
    <span class="feature-chip">${f.bathrooms} ba</span>
    <span class="feature-chip">${f.area_sqft} sq ft</span>
    <span class="feature-chip">${f.age_years} yrs old</span>
    <span class="feature-chip">${f.distance_km} km to center</span>
    <span class="feature-chip">${f.has_parking ? "Parking" : "No parking"}</span>
    <span class="feature-chip">${f.has_garden ? "Garden" : "No garden"}</span>
    <span class="feature-chip">Locality ${f.locality_score}/10</span>
  `;

  calcTbody.innerHTML = data.breakdown.map((row) => {
    const isBase = row.feature === "base_price";
    const valueCell = isBase ? "—" : `${row.value}${row.unit ? " " + row.unit : ""}`;
    const sign = row.contribution >= 0 ? "positive" : "negative";
    const contribText = isBase
      ? currency(row.contribution)
      : `${row.contribution >= 0 ? "+" : "−"}${currency(Math.abs(row.contribution))}`;
    return `<tr class="${sign}"><td>${row.label}</td><td>${valueCell}</td><td>${contribText}</td></tr>`;
  }).join("");

  calcTotal.textContent = currency(data.predicted_price);
  resultContent.hidden = false;
}

drawMap();
