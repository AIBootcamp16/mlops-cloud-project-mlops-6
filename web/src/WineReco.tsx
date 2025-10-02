// web/src/components/WineReco.tsx
import { useEffect, useMemo, useState } from "react";

/* ======================== Types ======================== */
type Style = "reds" | "whites" | "rose" | "port" | "sparkling";
type Screen = "search" | "results";
type Country = "" | "france" | "united states" | "italy" | "portugal" | "germany" | "spain";

interface Rating { average?: string | number; reviews?: string | number; }

interface RawCatalogItem {
  winery?: string;
  wine?: string;
  rating?: Rating;
  location?: string;
  image?: string;
  id: number | string;
}

interface Item {
  wine_id: number | string;
  wine_name: string;
  winery: string;
  label: string;
  rating: number;
  reviews: number;
  image_url?: string;
  country: string;
  region: string;
  style: Style;
  _rawScore?: number;
  score?: number;
  why?: string;
}

interface PredictResult {
  recommendations: Item[];
  model_version: string;
  inference_ms: number;
  country_hint: boolean;
  term_filtered: boolean;
  term_hint: boolean;
}

/* ================== Sample / External Sources ================== */
/** 작은 샘플 카탈로그(외부 JSON이 없어도 항상 동작) */
const CATALOG: RawCatalogItem[] = [
  { winery: "Maselva", wine: "Emporda 2012", rating: { average: "4.9", reviews: "88 ratings" }, location: "Spain · Empordà", image: "https://images.vivino.com/thumbs/ApnIiXjcT5Kc33OHgNb9dA_375x500.jpg", id: 1 },
  { winery: "Ernesto Ruffo", wine: "Amarone della Valpolicella Riserva N.V.", rating: { average: "4.9", reviews: "75 ratings" }, location: "Italy · Amarone della Valpolicella", image: "https://images.vivino.com/thumbs/nC9V6L2mQQSq0s-wZLcaxw_pb_x300.png", id: 2 },
  { winery: "Bodegas Emilio Moro", wine: "Ribera del Duero 2019", rating: { average: "4.6", reviews: "210 ratings" }, location: "Spain · Ribera del Duero", image: "https://images.unsplash.com/photo-1604908176997-43165108f7f0?q=80&w=1200", id: 3 },
  { winery: "Champagne Bollinger", wine: "Special Cuvée Brut N.V.", rating: { average: "4.5", reviews: "1,203 ratings" }, location: "France · Champagne", image: "https://images.unsplash.com/photo-1541976076758-347942db1970?q=80&w=1200", id: 4 },
  { winery: "Taylor's", wine: "Late Bottled Vintage Port 2017", rating: { average: "4.4", reviews: "540 ratings" }, location: "Portugal · Douro", image: "https://images.unsplash.com/photo-1514362545857-3bc16c4c76de?q=80&w=1200", id: 5 },
  { winery: "Robert Mondavi", wine: "Napa Valley Cabernet Sauvignon 2019", rating: { average: "4.3", reviews: "2,430 ratings" }, location: "United States · Napa Valley", image: "https://images.unsplash.com/photo-1547592180-85f173990554?q=80&w=1200", id: 6 },
  { winery: "Kistler", wine: "Sonoma Coast Chardonnay 2020", rating: { average: "4.4", reviews: "980 ratings" }, location: "United States · Sonoma Coast", image: "https://images.unsplash.com/photo-1622032287470-69eb7e0a80b2?q=80&w=1200", id: 7 },
];
const DEFAULT_CATALOG: RawCatalogItem[] = CATALOG;

const CATALOG_SOURCES: string[] = [
  "wines_reds.json",
  "wines_whites.json",
  "wines_rose.json",
  "wines_port.json",
  "wines_sparkling.json",
];

/* ======================== Helpers ======================== */
const TERM_ALIASES: Record<string, string[]> = {
  "샤도네이": ["chardonnay"],
  "피노 누아": ["pinot noir", "pinotnoir", "피노누아"],
  "피노누아": ["pinot noir", "pinotnoir", "피노 누아"],
  "까베르네 소비뇽": ["cabernet sauvignon", "cabernet"],
  "소비뇽 블랑": ["sauvignon blanc"],
  "리슬링": ["riesling"],
  "스파클링": ["sparkling", "champagne", "cava", "prosecco"],
  "포트": ["port"],
};

const normalizeText = (s: unknown): string =>
  String(s ?? "")
    .toLowerCase()
    .replace(/[-_]/g, " ")
    .replace(/\s+/g, " ")
    .trim();

const expandTerms = (arr: string[] = []): string[] => {
  const out = new Set<string>();
  for (const raw of arr) {
    const t = normalizeText(raw);
    if (!t) continue;
    out.add(t);
    const aliases = TERM_ALIASES[raw] || TERM_ALIASES[t] || [];
    for (const a of aliases) {
      const na = normalizeText(a);
      if (na) { out.add(na); out.add(na.replace(/\s+/g, "")); }
    }
    out.add(t.replace(/\s+/g, ""));
  }
  return Array.from(out);
};

const digitsOnlyToInt = (s: unknown): number => {
  const str = String(s ?? "");
  let digits = "";
  for (let i = 0; i < str.length; i++) {
    const ch = str[i];
    if (ch >= "0" && ch <= "9") digits += ch;
  }
  return digits ? parseInt(digits, 10) : 0;
};

const normalizeItem = (raw: RawCatalogItem): Item => {
  const parts = String(raw.location ?? "").split("·");
  const country = (parts[0] || "").replace(/\n/g, " ").trim().toLowerCase();
  const region  = (parts[1] || "").replace(/\n/g, " ").trim().toLowerCase();
  return {
    wine_id: raw.id,
    wine_name: String(raw.wine ?? ""),
    winery: String(raw.winery ?? ""),
    label: `${raw.winery ?? ""} ${raw.wine ?? ""}`.trim(),
    rating: parseFloat((raw.rating?.average as string | number | undefined)?.toString() ?? "0"),
    reviews: digitsOnlyToInt(raw.rating?.reviews),
    image_url: raw.image,
    country, region,
    style: "reds", // 이후 guessStyle로 갱신
  };
};

const upscaleImage = (url?: string): string => {
  if (!url) return "";
  if (url.includes("_pb_x300")) return url.replace("_pb_x300", "_pb_x600");
  if (/\d{3}x\d{3,4}\.(jpg|png)$/i.test(url)) {
    return url.replace(
      /(\d{3})x(\d{3,4})\.(jpg|png)$/i,
      (_m, w, h, ext) =>
        `${Math.max(600, parseInt(String(w)) * 2)}x${Math.max(800, parseInt(String(h)) * 2)}.${ext}`
    );
  }
  if (url.includes("images.unsplash.com") && !/([?&])w=/.test(url)) return `${url}&w=1200`;
  return url;
};

const guessStyle = (label: string, region: string): Style => {
  const t = (label + " " + region).toLowerCase();
  if (t.includes("champagne") || t.includes("cava") || t.includes("prosecco") || t.includes("sparkling")) return "sparkling";
  if (t.includes("port")) return "port";
  if (t.includes("rosé") || t.includes(" rose")) return "rose";
  if (t.includes("riesling") || t.includes("sauvignon blanc") || t.includes("chardonnay") || t.includes("white")) return "whites";
  return "reds";
};

const COUNTRY_OPTIONS: { value: Country; label: string }[] = [
  { value: "", label: "선택 안 함" },
  { value: "france", label: "France" },
  { value: "united states", label: "United States" },
  { value: "italy", label: "Italy" },
  { value: "portugal", label: "Portugal" },
  { value: "germany", label: "Germany" },
  { value: "spain", label: "Spain" },
];

const countryLabel = (val: Country): string =>
  COUNTRY_OPTIONS.find(o => o.value === val)?.label ?? "";

/* ======================== Component ======================== */
export default function WineRecoSearchAndResults() {
  const BRAND = { name: "WineReco", tagline: "Taste made personal" };
  const ACCENT = "#7B1733";

  // Screens
  const [screen, setScreen] = useState<Screen>("search");

  // Form state
  const [styles, setStyles] = useState<Style[]>(["sparkling"]);
  const [terms, setTerms] = useState<string[]>([]);
  const [preferCountry, setPreferCountry] = useState<Country>("");

  // Results
  const [result, setResult] = useState<PredictResult | null>(null);

  // Catalog
  const [catalog, setCatalog] = useState<RawCatalogItem[]>(DEFAULT_CATALOG);

  const mergeCatalogs = (arrays: unknown[]): RawCatalogItem[] => {
    const out: RawCatalogItem[] = [];
    const seen = new Set<number | string>();
    for (const arr of arrays) {
      if (!Array.isArray(arr)) continue;
      for (const x of arr as RawCatalogItem[]) {
        const id = (x?.id ?? `${x?.winery}-${x?.wine}`) as number | string;
        if (id == null || seen.has(id)) continue;
        seen.add(id);
        out.push(x);
      }
    }
    return out.length ? out : DEFAULT_CATALOG;
  };

  // Load external JSONs (optional)
  useEffect(() => {
    let cancelled = false;
    (async () => {
      try {
        const fetched: unknown[] = [];
        for (const path of CATALOG_SOURCES) {
          try {
            const res = await fetch(path);
            if (res.ok) {
              const data = await res.json();
              if (Array.isArray(data)) fetched.push(data);
            }
          } catch { /* ignore per-source errors */ }
        }
        const merged = mergeCatalogs(fetched);
        if (!cancelled) setCatalog(merged);
      } catch {
        if (!cancelled) setCatalog(DEFAULT_CATALOG);
      }
    })();
    return () => { cancelled = true; };
  }, []);

  // Normalize & infer style
  const items: Item[] = useMemo(
    () => catalog.map(normalizeItem).map(it => ({ ...it, style: guessStyle(it.label, it.region) })),
    [catalog]
  );

  const scoreRaw = (it: Item): number => {
    const quality = (it.rating / 5) * Math.log1p(it.reviews);
    let boost = 0;
    if (styles.includes(it.style)) boost += 0.4;
    if (preferCountry && it.country === preferCountry.toLowerCase()) boost += 0.2;
    if (terms.length) {
      const joined = normalizeText(it.label + " " + it.region);
      const joinedNoSpace = joined.replace(/\s+/g, "");
      const needles = expandTerms(terms);
      const hit = needles.some(t => joined.includes(t) || joinedNoSpace.includes(t));
      if (hit) boost += 0.35;
    }
    return Math.max(0, quality + boost);
  };

  const buildWhy = (it: Item): string => {
    const bits: string[] = [];
    if (styles.includes(it.style)) bits.push("선호 스타일");
    if (preferCountry && it.country === preferCountry.toLowerCase()) bits.push("선호 국가");
    if (it.rating) bits.push(`★${it.rating.toFixed(1)}`);
    if (it.reviews) bits.push(`${it.reviews} reviews`);
    return bits.join(" · ") || "추천";
  };

  const matchTerm = (it: Item, needles: string[]): boolean => {
    if (!needles || !needles.length) return true;
    const joined = normalizeText(it.label + " " + it.region);
    const joinedNoSpace = joined.replace(/\s+/g, "");
    return needles.some(t => joined.includes(t) || joinedNoSpace.includes(t));
  };

  const predict = (): PredictResult => {
    const needles = expandTerms(terms);

    // 1) style pool
    const base = items.filter(it => styles.includes(it.style));

    // 2) hard filter by terms
    let selected = base;
    let termFiltered = false;
    if (needles.length) {
      termFiltered = true;
      selected = base.filter(it => matchTerm(it, needles));
    }

    // 3) score & sort
    let scored = selected
      .map(it => ({ ...it, _rawScore: scoreRaw(it) }))
      .sort((a, b) => (b._rawScore! - a._rawScore!));

    // 4) top3 + backfill
    let top = scored.slice(0, 3);
    let countryHint = false;
    let termHint = false;

    if (top.length < 3) {
      termHint = needles.length > 0;
      const backfillPool = base
        .filter(it => !selected.includes(it))
        .map(it => ({ ...it, _rawScore: scoreRaw(it) }))
        .sort((a, b) => (b._rawScore! - a._rawScore!));
      top = top.concat(backfillPool.slice(0, 3 - top.length));
    }

    // 5) preferred country guarantee within style pool
    if (preferCountry) {
      const want = preferCountry.toLowerCase();
      const poolPref = scored.filter(it => it.country === want);
      if (poolPref.length > 0 && !top.some(it => it.country === want)) {
        const bestPreferred = poolPref[0];
        const replaced = [...top.slice(0, 2), bestPreferred].sort((a, b) => (b._rawScore! - a._rawScore!));
        top = replaced.slice(0, 3);
      } else if (poolPref.length === 0) {
        countryHint = true;
      }
    }

    const maxScore = Math.max(...top.map(x => x._rawScore || 0), 1);
    const recos: Item[] = top.map(it => ({
      ...it,
      score: (it._rawScore || 0) / maxScore,
      why: buildWhy(it),
    }));

    return {
      recommendations: recos,
      model_version: "wine-reco@Production#v7",
      inference_ms: 12,
      country_hint: countryHint,
      term_filtered: termFiltered,
      term_hint: termHint,
    };
  };

  const onSubmit = (e?: React.FormEvent | React.MouseEvent) => {
    e?.preventDefault?.();
    setResult(predict());
    setScreen("results");
  };

  /* ======================== UI ======================== */
  return (
    <div className="min-h-screen bg-white text-neutral-900">
      {/* Header */}
      <header className="max-w-6xl mx-auto px-6 py-5 flex items-center justify-between">
        <div className="flex items-center gap-3">
          <div className="w-9 h-9 rounded-xl text-white flex items-center justify-center font-semibold" style={{ backgroundColor: ACCENT }}>WR</div>
          <div>
            <h1 className="text-2xl font-semibold tracking-tight">{BRAND.name}</h1>
            <p className="text-xs text-neutral-500">{BRAND.tagline}</p>
          </div>
        </div>
        {screen === "results" && result && (
          <div className="flex items-center gap-2 text-xs">
            <span className="px-2 py-1 rounded-full bg-neutral-900/5 text-neutral-700 border border-neutral-900/10">{result.model_version}</span>
            <span className="px-2 py-1 rounded-full bg-neutral-900/5 border border-neutral-900/10">API {result.inference_ms} ms</span>
          </div>
        )}
      </header>

      {/* Search Screen */}
      {screen === "search" && (
        <main className="max-w-6xl mx-auto px-6">
          <div className="rounded-2xl border border-neutral-200 bg-white p-6 shadow-sm">
            <h2 className="text-lg font-semibold">당신만의 와인을 큐레이팅합니다</h2>
            <p className="mt-1 text-sm text-neutral-500">취향을 읽고 오늘의 추천을 전해드려요.</p>

            <form className="mt-5 grid grid-cols-1 gap-5" onSubmit={onSubmit}>
              <div className="grid grid-cols-1 md:grid-cols-12 gap-3 items-end">
                <div className="md:col-span-6">
                  <label className="text-sm text-neutral-600">스타일</label>
                  <div className="mt-2 flex flex-wrap gap-2">
                    {(["reds", "whites", "rose", "port", "sparkling"] as Style[]).map((s) => {
                      const active = styles.includes(s);
                      return (
                        <button
                          type="button"
                          key={s}
                          onClick={() => setStyles(prev => (prev.includes(s) ? prev.filter(x => x !== s) : [...prev, s]))}
                          className={`px-3 py-2 rounded-full whitespace-nowrap border text-sm ${active ? "text-white" : "text-neutral-700"}`}
                          style={active ? { backgroundColor: ACCENT, borderColor: ACCENT } : { borderColor: "#e5e5e5" }}
                        >
                          {s}
                        </button>
                      );
                    })}
                  </div>
                </div>

                <div className="md:col-span-4">
                  <label className="text-sm text-neutral-600">선호 국가</label>
                  <select
                    className="mt-2 w-full rounded-xl bg-white border border-neutral-300 px-3 py-2"
                    value={preferCountry}
                    onChange={(e) => setPreferCountry(e.target.value as Country)}
                  >
                    {COUNTRY_OPTIONS.map(opt => (
                      <option key={opt.value || "none"} value={opt.value}>{opt.label}</option>
                    ))}
                  </select>
                </div>

                <div className="md:col-span-2 flex md:justify-end">
                  <button type="submit" className="mt-2 px-5 py-2 rounded-xl text-white text-sm shadow-sm w-full md:w-auto" style={{ backgroundColor: ACCENT }}>
                    추천 받기
                  </button>
                </div>
              </div>

              <div>
                <label className="text-sm text-neutral-600">검색어(생략 가능)</label>
                <input
                  className="mt-2 w-full rounded-xl bg-white border border-neutral-300 px-3 py-2 placeholder-neutral-400 focus:outline-none focus:ring-2"
                  style={{ outlineColor: ACCENT }}
                  placeholder="chardonnay, pinot noir / 샤도네이, 피노 누아"
                  value={terms.join(", ")}
                  onChange={(e) =>
                    setTerms(
                      e.target.value
                        .split(",")
                        .map(s => s.trim().toLowerCase())
                        .filter(Boolean)
                    )
                  }
                />
              </div>
            </form>
          </div>
        </main>
      )}

      {/* Results Screen */}
      {screen === "results" && result && (
        <main className="max-w-6xl mx-auto px-6">
          <section className="rounded-2xl border border-neutral-200 bg-white p-4 shadow-sm">
            <div className="flex items-center justify-between mb-3">
              <div className="flex items-center gap-2">
                <h2 className="text-lg font-semibold">추천 결과 Top 3</h2>
                {result.term_filtered && (
                  <span className="text-xs px-2 py-1 rounded-full" style={{ backgroundColor: "#F3E8EE", color: "#7B1733", border: "1px solid #E9D5DF" }}>
                    키워드 필터 적용
                  </span>
                )}
              </div>
              <div className="text-xs text-neutral-500">
                {styles.join(", ")}
                {preferCountry ? ` · ${countryLabel(preferCountry)}` : ""}
                {terms.length ? ` · ${terms.join(", ")}` : ""}
              </div>
            </div>

            {(result.country_hint || result.term_hint) && (
              <div className="mb-3 text-sm text-amber-700 bg-amber-50 border border-amber-200 rounded-xl px-3 py-2">
                {result.term_hint
                  ? "입력한 키워드와 정확히 매칭되는 항목이 부족해요. 키워드를 줄이거나 스타일을 넓혀보세요!"
                  : "선택한 와인 종류에 선호 국가의 데이터가 부족해요. 와인 종류를 넓혀보세요!"}
              </div>
            )}

            <div className="grid grid-cols-1 gap-4">
              {result.recommendations.map((it, idx) => (
                <article key={String(it.wine_id)} className="rounded-2xl border border-neutral-200 bg-white p-4">
                  <div className="flex items-start justify-between gap-3">
                    <div className="flex gap-3">
                      <img
                        src={upscaleImage(it.image_url) || "https://placehold.co/600x800?text=Wine"}
                        alt={it.label}
                        referrerPolicy="no-referrer"
                        onError={(e) => {
                          const img = e.currentTarget;
                          img.onerror = null;
                          img.src = "https://placehold.co/600x800?text=Wine";
                        }}
                        className="w-40 h-56 object-cover rounded-xl border border-neutral-200"
                        width={800}
                        height={1120}
                      />
                      <div>
                        <div className="text-sm text-neutral-500">#{idx + 1}</div>
                        <h3 className="text-xl font-semibold mt-1 leading-tight">{it.label}</h3>
                        <div className="mt-1 text-sm text-neutral-600">
                          {it.winery} · {countryLabel(it.country as Country)}
                        </div>
                      </div>
                    </div>
                    <div className="text-right">
                      <div className="text-sm text-neutral-500">Rating</div>
                      <div className="text-lg font-semibold">{it.rating ? `${it.rating.toFixed(1)}★` : "-"}</div>
                      <div className="text-xs text-neutral-500">{it.reviews || 0} reviews</div>
                    </div>
                  </div>

                  <div className="mt-4 flex gap-2">
                    <button className="px-3 py-2 rounded-xl border border-neutral-300 text-sm">Details</button>
                    <button
                      onClick={(e) => { const btn = e.currentTarget; btn.disabled = true; btn.querySelector("span")!.innerHTML = "Saved"; }}
                      className="px-3 py-2 rounded-xl text-white text-sm inline-flex items-center gap-2"
                      style={{ backgroundColor: ACCENT }}
                    >
                      <svg width="18" height="18" viewBox="0 0 24 24" fill={ACCENT} stroke="white" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                        <path d="M20.84 4.61a5.5 5.5 0 0 0-7.78 0L12 5.67l-1.06-1.06a5.5 5.5 0 0 0-7.78 7.78l1.06 1.06L12 21.23l7.78-7.78 1.06-1.06a5.5 5.5 0 0 0 0-7.78z"></path>
                      </svg>
                      <span>Save</span>
                    </button>
                    <button onClick={(e) => { e.currentTarget.innerHTML = "Dismissed"; (e.currentTarget as HTMLButtonElement).disabled = true; }} className="px-3 py-2 rounded-xl border border-neutral-300 text-sm">
                      Not for me
                    </button>
                  </div>
                </article>
              ))}
            </div>

            <div className="mt-6 flex items-center justify-between">
              <div className="text-xs text-neutral-500">
                Model Version:{" "}
                <span className="px-2 py-1 rounded-full bg-neutral-900/5 text-neutral-700 border border-neutral-900/10">wine-reco@Production#v7</span>
              </div>
              <button onClick={() => setScreen("search")} className="px-3 py-2 rounded-xl text-white text-sm shadow-sm" style={{ backgroundColor: ACCENT }}>
                다시 찾기
              </button>
            </div>
          </section>
        </main>
      )}
    </div>
  );
}
