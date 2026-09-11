import express from 'express';
import path from 'path';
import { createServer as createViteServer } from 'vite';
import dotenv from 'dotenv';
import { GoogleGenAI } from '@google/genai';
import { DEMO_DATASETS, DEMO_MATERIALS_LIST } from './src/data/demoMaterials';
import { DiscoveryAnalysisResult } from './src/types';

dotenv.config();

function getGeminiClient(): GoogleGenAI | null {
  const apiKey = process.env.GEMINI_API_KEY;
  if (!apiKey || apiKey === 'MY_GEMINI_API_KEY' || apiKey.trim() === '') {
    return null;
  }
  return new GoogleGenAI({
    apiKey,
    httpOptions: {
      headers: {
        'User-Agent': 'aistudio-build',
      },
    },
  });
}

async function startServer() {
  const app = express();
  const PORT = 3000;

  // Middleware for JSON body parsing (with large limit for base64 images)
  app.use(express.json({ limit: '30mb' }));

  // --- API Endpoints ---

  // Health check
  app.get('/api/health', (req, res) => {
    res.json({
      status: 'ok',
      hasGeminiKey: Boolean(process.env.GEMINI_API_KEY && process.env.GEMINI_API_KEY !== 'MY_GEMINI_API_KEY'),
      timestamp: new Date().toISOString(),
    });
  });

  // Demo materials list
  app.get('/api/demo-materials', (req, res) => {
    res.json(DEMO_MATERIALS_LIST);
  });

  // Retrieve single demo dataset directly
  app.get('/api/demo-materials/:key', (req, res) => {
    const materialKey = req.params.key;
    const dataset = DEMO_DATASETS[materialKey];
    if (dataset) {
      res.json(dataset);
    } else {
      res.status(404).json({ error: `Demo material '${materialKey}' not found.` });
    }
  });

  // Analyze material (Vision + Biology + Natural Products + Targets + Pharmacology + Safety + Opportunities)
  app.post('/api/analyze', async (req, res) => {
    try {
      const {
        imageBase64,
        imageMimeType = 'image/jpeg',
        userContext,
        demoMaterialKey,
        imageFileName,
        imageFileSize,
      } = req.body;

      // If explicitly requested demo dataset or demo material key specified
      if (demoMaterialKey && DEMO_DATASETS[demoMaterialKey]) {
        const baseData = JSON.parse(JSON.stringify(DEMO_DATASETS[demoMaterialKey])) as DiscoveryAnalysisResult;
        if (imageBase64) {
          baseData.imagePreviewUrl = imageBase64;
          baseData.imageFileName = imageFileName || baseData.imageFileName;
          baseData.imageFileSize = imageFileSize || baseData.imageFileSize;
        }
        if (userContext && userContext.trim()) {
          baseData.userContext = userContext.trim();
        }
        return res.json(baseData);
      }

      const ai = getGeminiClient();

      // If Gemini is available and an image or context was provided, attempt AI analysis
      if (ai && (imageBase64 || userContext)) {
        try {
          const parts: any[] = [];

          if (imageBase64) {
            // strip data url prefix if present
            const cleanBase64 = imageBase64.replace(/^data:image\/[a-zA-Z0-9+.-]+;base64,/, '');
            parts.push({
              inlineData: {
                data: cleanBase64,
                mimeType: imageMimeType,
              },
            });
          }

          const systemPrompt = `You are Bio2Drug AI, a specialized natural-product drug discovery and target prioritization platform.
Your mission: Given an image or contextual description of a biological material or agricultural waste (plant, fruit, fungus, marine organism, agricultural biomass), identify candidate natural products, molecular targets, pharmacological relevance, safety/ADME screening, and formulate a prioritized research hypothesis.

CRITICAL SCIENTIFIC SAFEGUARDS:
- Do NOT claim the image definitively identifies a drug.
- Do NOT claim this proves therapeutic efficacy or cures diseases.
- Position all findings as "AI-assisted visual identification" and "prioritized research opportunities requiring experimental validation".
- Return ONLY a valid JSON object strictly matching the structure provided below.

JSON SCHEMA:
{
  "source": {
    "common_name": string,
    "scientific_name": string,
    "confidence": number (integer 60-95),
    "category": string,
    "material_part": string,
    "alternative_candidates": string[],
    "visual_features": string[],
    "biomass_origin": string
  },
  "compounds": [
    {
      "id": string,
      "name": string,
      "chemical_class": string,
      "formula": string,
      "mw": number,
      "source_organism": string,
      "activities": string[],
      "evidence_level": "Preclinical (In vivo & In vitro)" | "In vitro & Enzymatic" | "In silico & Mechanistic",
      "potential_targets": string[],
      "pubchem_cid": string,
      "chembl_id": string,
      "summary": string
    }
  ],
  "targets": [
    {
      "id": string,
      "target_name": string,
      "gene_symbol": string,
      "target_class": string,
      "associated_pathway": string,
      "disease_area": string,
      "evidence_strength": "Strong (Biochemical + Cellular)" | "Moderate (Literature association)" | "Putative (Docking prediction)",
      "interaction_type": string,
      "uniprot_id": string,
      "pdb_id": string,
      "linked_compound_ids": string[]
    }
  ],
  "pharmacology": [
    {
      "application_area": string,
      "research_relevance": string,
      "target_pathways": string[],
      "therapeutic_hypothesis": string,
      "readout_assays": string[]
    }
  ],
  "safety": [
    {
      "compound_id": string,
      "compound_name": string,
      "signals": {
        "toxicity": { "status": "favorable" | "uncertain" | "concern" | "insufficient", "detail": string },
        "cytotoxicity": { "status": "favorable" | "uncertain" | "concern" | "insufficient", "detail": string },
        "hepatotoxicity": { "status": "favorable" | "uncertain" | "concern" | "insufficient", "detail": string },
        "mutagenicity": { "status": "favorable" | "uncertain" | "concern" | "insufficient", "detail": string }
      },
      "adme": {
        "drug_likeness": { "status": "favorable" | "uncertain" | "concern" | "insufficient", "detail": string },
        "solubility": { "status": "favorable" | "uncertain" | "concern" | "insufficient", "detail": string },
        "absorption": { "status": "favorable" | "uncertain" | "concern" | "insufficient", "detail": string },
        "distribution": { "status": "favorable" | "uncertain" | "concern" | "insufficient", "detail": string },
        "metabolism": { "status": "favorable" | "uncertain" | "concern" | "insufficient", "detail": string },
        "excretion": { "status": "favorable" | "uncertain" | "concern" | "insufficient", "detail": string }
      }
    }
  ],
  "evidence": {
    "overall_score": number (0-100),
    "natural_product_evidence": number (0-100),
    "target_evidence": number (0-100),
    "pharmacology_evidence": number (0-100),
    "safety_evidence": number (0-100),
    "sources": [
      {
        "id": string,
        "database_name": string,
        "title": string,
        "identifier": string,
        "evidence_type": "Peer-reviewed Journal" | "Structural DB (PDB/UniProt)" | "Bioactivity Repository (ChEMBL)" | "Natural Products DB (NPASS)",
        "year": number,
        "strength": "High" | "Medium" | "Preliminary",
        "url": string
      }
    ]
  },
  "opportunities": [
    {
      "id": string,
      "rank": number,
      "title": string,
      "priority_score": number (0-100),
      "score_breakdown": {
        "natural_product": number,
        "target_evidence": number,
        "pharmacological_relevance": number,
        "safety_profile": number,
        "novelty_gap": number
      },
      "primary_compound": string,
      "primary_target": string,
      "disease_domain": string,
      "why_it_matters": string,
      "major_uncertainty": string,
      "supporting_sources": string[]
    }
  ],
  "research_hypothesis": {
    "title": string,
    "research_question": string,
    "hypothesis": string,
    "biological_source": string,
    "candidate_compounds": string[],
    "molecular_targets": string[],
    "proposed_mechanism": {
      "compound": string,
      "target": string,
      "pathway": string,
      "biological_effect": string,
      "narrative": string
    },
    "research_gap": {
      "known_evidence": string[],
      "proposed_hypothesis": string[],
      "unknown_gaps": string[]
    },
    "experimental_validation": [
      { "stage": string, "technique": string, "objective": string, "readout": string }
    ],
    "expected_outcome": string,
    "limitations": string[],
    "references": string[]
  }
}`;

          parts.push({
            text: `Analyze this biological specimen/biomass material.
User Context notes provided by researcher: "${userContext || 'No additional notes provided.'}"
Identify the botanical or biological taxon, candidate natural product compounds, molecular targets, pharmacological relevance, safety/ADME profile, and prioritize a research hypothesis.
Remember to return strictly JSON.`
          });

          const response = await ai.models.generateContent({
            model: 'gemini-3.8-flash',
            contents: { parts },
            config: {
              systemInstruction: systemPrompt,
              responseMimeType: 'application/json',
              temperature: 0.2,
            },
          });

          const rawText = response.text || '{}';
          const parsed = JSON.parse(rawText);

          const result: DiscoveryAnalysisResult = {
            id: 'res-ai-' + Date.now(),
            isDemoDataset: false,
            userContext: userContext || undefined,
            imagePreviewUrl: imageBase64,
            imageFileName: imageFileName || 'uploaded_biomass_sample.jpg',
            imageFileSize: imageFileSize || (imageBase64 ? Math.round(imageBase64.length * 0.75) : undefined),
            source: parsed.source,
            compounds: parsed.compounds || [],
            targets: parsed.targets || [],
            pharmacology: parsed.pharmacology || [],
            safety: parsed.safety || [],
            evidence: parsed.evidence || {
              overall_score: 80,
              natural_product_evidence: 85,
              target_evidence: 78,
              pharmacology_evidence: 82,
              safety_evidence: 75,
              sources: [],
            },
            opportunities: parsed.opportunities || [],
            research_hypothesis: {
              ...parsed.research_hypothesis,
              generated_timestamp: new Date().toISOString(),
              is_ai_generated: true,
            },
          };

          return res.json(result);
        } catch (geminiError: any) {
          console.warn('Gemini live analysis encountered an issue, falling back to curated dataset:', geminiError?.message);
          // Fall through to fallback
        }
      }

      // Fallback to high-quality curated dataset if no Gemini key or upon error
      // If userContext mentions keywords like turmeric, tea, citrus, rice, neem, reishi, or mango
      let matchedKey = 'mango_peel';
      const queryText = ((userContext || '') + ' ' + (imageFileName || '')).toLowerCase();
      if (queryText.includes('turmeric') || queryText.includes('curcuma')) matchedKey = 'turmeric_waste';
      else if (queryText.includes('tea') || queryText.includes('camellia')) matchedKey = 'spent_tea';
      else if (queryText.includes('citrus') || queryText.includes('orange') || queryText.includes('lemon')) matchedKey = 'citrus_peel';
      else if (queryText.includes('rice') || queryText.includes('straw') || queryText.includes('cereal')) matchedKey = 'rice_straw';
      else if (queryText.includes('neem') || queryText.includes('azadirachta')) matchedKey = 'neem_biomass';
      else if (queryText.includes('mushroom') || queryText.includes('reishi') || queryText.includes('fungus')) matchedKey = 'reishi_fungal';

      const fallbackData = JSON.parse(JSON.stringify(DEMO_DATASETS[matchedKey])) as DiscoveryAnalysisResult;
      fallbackData.isDemoDataset = true;
      fallbackData.demoMaterialKey = matchedKey;
      if (imageBase64) fallbackData.imagePreviewUrl = imageBase64;
      if (userContext) fallbackData.userContext = userContext;
      if (imageFileName) fallbackData.imageFileName = imageFileName;
      if (imageFileSize) fallbackData.imageFileSize = imageFileSize;

      return res.json(fallbackData);
    } catch (err: any) {
      console.error('Analysis error:', err);
      res.status(500).json({ error: 'Failed to complete discovery analysis', details: err?.message });
    }
  });

  // Generate alternative research hypothesis
  app.post('/api/generate-hypothesis', async (req, res) => {
    try {
      const { opportunity, source, compoundName, targetName, userNotes } = req.body;
      const ai = getGeminiClient();

      if (ai) {
        try {
          const prompt = `Generate a rigorous, 12-section scientific research hypothesis proposal for natural-product drug discovery.
Source material: ${source?.common_name} (${source?.scientific_name}), part: ${source?.material_part}
Opportunity focus: ${opportunity?.title || compoundName + ' targeting ' + targetName}
Candidate Compound: ${compoundName || opportunity?.primary_compound}
Molecular Target: ${targetName || opportunity?.primary_target}
Researcher Custom Notes: "${userNotes || 'Standard grant-application proposal'}"

Return ONLY a valid JSON object matching this schema:
{
  "title": string,
  "research_question": string,
  "hypothesis": string,
  "biological_source": string,
  "candidate_compounds": string[],
  "molecular_targets": string[],
  "proposed_mechanism": {
    "compound": string,
    "target": string,
    "pathway": string,
    "biological_effect": string,
    "narrative": string
  },
  "research_gap": {
    "known_evidence": string[],
    "proposed_hypothesis": string[],
    "unknown_gaps": string[]
  },
  "experimental_validation": [
    { "stage": string, "technique": string, "objective": string, "readout": string }
  ],
  "expected_outcome": string,
  "limitations": string[],
  "references": string[]
}`;

          const response = await ai.models.generateContent({
            model: 'gemini-3.8-flash',
            contents: prompt,
            config: {
              responseMimeType: 'application/json',
              temperature: 0.3,
            },
          });

          const parsed = JSON.parse(response.text || '{}');
          return res.json({
            ...parsed,
            generated_timestamp: new Date().toISOString(),
            is_ai_generated: true,
          });
        } catch (e: any) {
          console.warn('Gemini hypothesis generation failed, falling back to structured generator:', e?.message);
        }
      }

      // Deterministic scientific hypothesis generator
      const comp = compoundName || opportunity?.primary_compound || 'Bioactive phytochemical';
      const tgt = targetName || opportunity?.primary_target || 'Regulatory signaling target';
      const sci = source?.scientific_name || 'natural biological biomass';

      const structuredHypothesis = {
        title: `Investigation of ${comp} from Discarded ${source?.common_name || 'Biomass'} as a Modulator of ${tgt}`,
        research_question: `Does targeted green extraction of ${source?.material_part || 'processing waste'} yield stable ${comp} capable of modulating ${tgt} signaling without cytotoxic liabilities?`,
        hypothesis: `Standardized ${comp} derived from agro-waste of ${sci} exhibits selective binding affinity toward ${tgt}, attenuating pathological pathway hyperactivity while maintaining physiological homeostatic signaling in cellular models.`,
        biological_source: `Upcycled biomass of ${source?.common_name || 'specimen'} (${sci}), focusing on secondary metabolites accumulated in peel or lignocellulose matrices.`,
        candidate_compounds: [comp, 'Related secondary congeners', 'Deglycosylated bio-active metabolites'],
        molecular_targets: [tgt, 'Downstream transcriptional effectors', 'Enzymatic regulators'],
        proposed_mechanism: {
          compound: comp,
          target: tgt,
          pathway: opportunity?.disease_domain || 'Intracellular regulatory cascade',
          biological_effect: 'Suppression of pathological disease mediators and restoration of cellular homeostasis',
          narrative: `${comp} selectively occupies active or allosteric cavities on ${tgt}, preventing substrate phosphorylation or transcriptional complex recruitment.`
        },
        research_gap: {
          known_evidence: [
            `Preliminary literature associates ${comp} with biological activities in general screening assays.`,
            `The raw biological source ${sci} contains high concentrations of secondary defense metabolites.`
          ],
          proposed_hypothesis: [
            `Modern subcritical or ultrasonic extraction recovers intact active chemical scaffolds without thermal decomposition.`,
            `Direct biophysical binding to ${tgt} occurs with micromolar to nanomolar affinity constants.`
          ],
          unknown_gaps: [
            `Kinetic binding mode (competitive vs non-competitive allosteric) has not been experimentally resolved by structural biology.`,
            `Bioavailability and metabolic stability in standard physiological transport models remain to be confirmed.`
          ]
        },
        experimental_validation: [
          {
            stage: '1. Upcycling Green Extraction & Purification',
            technique: 'Ultrasound-assisted green solvent extraction and preparative HPLC',
            objective: `Isolate >95% pure ${comp} from dry biomass feedstock.`,
            readout: 'Yield percentage, HPLC-DAD chromatographic purity confirmation.'
          },
          {
            stage: '2. Target Affinity & Binding Kinetics',
            technique: 'Surface Plasmon Resonance (SPR) or Microscale Thermophoresis (MST)',
            objective: `Quantify binding dissociation constant (KD) against recombinant ${tgt}.`,
            readout: 'KD equilibrium constant, association (ka) and dissociation (kd) rates.'
          },
          {
            stage: '3. In Vitro Cellular Pathway Validation',
            technique: 'Dual-luciferase reporter gene assay and cytokine Western blotting',
            objective: `Measure functional inhibition of downstream pathway transcription.`,
            readout: 'Dose-response IC50 curve, relative luminescence units, target protein phosphorylation levels.'
          },
          {
            stage: '4. Cellular Safety & ADME Profiling',
            technique: 'CCK-8 viability assay across primary human cell lines and human liver microsomes (HLM)',
            objective: 'Confirm favorable therapeutic safety index and clearance kinetics.',
            readout: 'IC50/CC50 therapeutic ratio, metabolic half-life (t1/2), CYP450 inhibition profile.'
          }
        ],
        expected_outcome: `Establishment of a rigorous evidence-backed research pipeline validating ${comp} from ${source?.common_name || 'biomass'} as a prioritized hit for medicinal chemistry optimization.`,
        limitations: [
          'High-throughput screening hits require orthogonal validation to rule out pan-assay interference (PAINS).',
          'Batch-to-batch variation in natural biomass raw materials requires strict chemical standardization.'
        ],
        references: [
          'ChEMBL Bioactivity Database record for candidate compound series',
          'Protein Data Bank (PDB) structural coordinates for target family',
          'PubChem Compound repository documentation'
        ],
        generated_timestamp: new Date().toISOString(),
        is_ai_generated: false,
      };

      return res.json(structuredHypothesis);
    } catch (err: any) {
      res.status(500).json({ error: 'Failed to generate research hypothesis', details: err?.message });
    }
  });

  // --- Vite Middleware / Static Serving ---
  if (process.env.NODE_ENV !== 'production') {
    const vite = await createViteServer({
      server: { middlewareMode: true },
      appType: 'spa',
    });
    app.use(vite.middlewares);
  } else {
    const distPath = path.join(process.cwd(), 'dist');
    app.use(express.static(distPath));
    app.get('*', (req, res) => {
      res.sendFile(path.join(distPath, 'index.html'));
    });
  }

  app.listen(PORT, '0.0.0.0', () => {
    console.log(`🌿 Bio2Drug AI Server listening on http://0.0.0.0:${PORT}`);
  });
}

startServer();
