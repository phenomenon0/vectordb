// All initial reactive state for the DeepData app
export function initialState() {
  return {
    // Navigation
    page: 'dashboard',
    pages: [
      { id: 'dashboard', label: 'Dashboard' },
      { id: 'data', label: 'Data' },
      { id: 'settings', label: 'Settings' },
      { id: 'chat', label: 'Chat' },
      { id: 'admin', label: 'Admin' },
    ],
    dataSection: 'search',
    showCosts: false,
    showFeedback: false,
    apiBase: window.location.origin,
    toasts: [],
    requestLog: [],

    // Onboarding
    showOnboarding: !localStorage.getItem('deepdata_onboarded'),
    onboardingStep: 0,
    onboardingSteps: [
      { id: 'welcome', title: 'Welcome to DeepData' },
      { id: 'mode', title: 'Choose Mode' },
      { id: 'embedder', title: 'Configure Embedder' },
      { id: 'collection', title: 'First Collection' },
    ],
    onboardingMode: 'local',
    onboardingEmbedderType: 'ollama',
    onboardingEmbedderUrl: '',
    onboardingApiKey: '',
    onboardingCollName: '',
    onboardingCollFields: '[]',
    onboardingComplete: false,

    // Dashboard
    health: {},
    mode: {},
    indexes: {},
    uptime: '',
    startTime: null,
    dashIntegrityResult: null,
    dashCosts: {},
    dashFeedback: {},

    // Monitoring
    metrics: {},
    metricsLoaded: false,
    metricsRefreshId: null,
    indexTypes: [],
    indexDetailedStats: {},

    // Search
    searchQuery: '',
    searchCollection: '',
    searchTopK: 10,
    searchMode: 'vector',
    searchAlpha: 0.5,
    searchMeta: '',
    showFilters: false,
    searching: false,
    searchResults: [],
    searchLatency: null,

    // Hybrid search params
    hybridStrategy: 'rrf',
    hybridWeightDense: 0.7,
    hybridWeightSparse: 0.3,

    // Recommend
    recommendPositiveIds: '',
    recommendNegativeIds: '',
    recommendNegativeWeight: 0.5,
    recommendTopK: 10,
    recommendResults: [],
    recommending: false,

    // Discover
    discoverTargetId: '',
    discoverContextPairs: [],
    discoverResults: [],
    discovering: false,

    // Geo filter
    geoFilterEnabled: false,
    geoFilterMode: 'radius',
    geoFilterLat: '',
    geoFilterLon: '',
    geoFilterDistanceKm: 10,
    geoFilterTopLeft: '',
    geoFilterBottomRight: '',

    // Explorer
    explorerDocs: [],
    explorerCollection: 'default',
    explorerOffset: 0,
    explorerLimit: 50,
    explorerTotal: 0,

    // Collections
    v2Collections: [],
    newCollName: '',
    newCollFields: '',

    // Costs
    costs: {},
    dailyCosts: [],

    // Feedback
    feedbackStats: {},
    fbInteractionId: '',
    fbType: 'explicit',
    fbRating: 1,
    fbText: '',
    intQuery: '',
    intResults: '',
    intCollection: 'default',

    // Settings
    integrityResult: null,
    integrityRunning: false,
    compactResult: null,
    compactRunning: false,
    importRunning: false,
    newIdxCollection: 'default',
    newIdxType: 'hnsw',
    newIdxConfig: '{}',
    showTools: false,
    showEmbedderConfig: false,
    showApiKeys: false,

    // Embedder config
    embType: '',
    embModel: '',
    embUrl: '',
    embKey: '',
    embDim: 0,
    embSaving: false,
    embTestResult: null,

    // API Keys
    apiKeysSet: {},
    apiKeyInputs: { openai: '', deepseek: '', anthropic: '', openrouter: '', cerebras: '' },
    apiKeysSaving: false,

    // Tools
    embedText: '',
    embedResult: '',
    embedRunning: false,
    extractText: '',
    extractTypes: '',
    extractResult: '',
    extractRunning: false,
    batchCollection: 'default',
    batchText: '',
    batchId: '',
    batchMeta: '',

    // LLM Provider
    llmProvider: localStorage.getItem('llmProvider') || 'ollama',
    llmUrl: localStorage.getItem('llmUrl') || 'http://localhost:11434/v1',
    llmApiKey: localStorage.getItem('llmApiKey') || '',
    llmModel: localStorage.getItem('llmModel') || '',
    llmModels: [],
    llmConnected: false,
    llmConnecting: false,
    llmNeedsKey: false,

    // Chat
    chatMessages: [],
    chatInput: '',
    chatStreaming: false,
    chatStreamContent: '',
    chatAbortCtrl: null,
    chatSystemPromptOverride: '',
    showChatSystem: false,
    chatCollection: '',

    // Viewer
    viewerOpen: false,
    viewerResult: null,
    viewerContent: null,
    viewerModality: null,
    viewerLoading: false,
    viewerIndex: -1,

    // Image viewer
    imgScale: 1, imgX: 0, imgY: 0, imgDragging: false, imgLastX: 0, imgLastY: 0,

    // Transcription
    transcribing: false,
    transcription: '',

    // Annotations
    annotations: {},
    annotationMode: null,
    activeAnnotation: null,
    showAnnotations: false,
    annotationNote: '',
    annotationColor: '#3b82f6',

    // Knowledge Graph
    kgInputText: '',
    kgEntityTypes: [],
    kgBatchMode: false,
    kgTemporalMode: false,
    kgEntities: [],
    kgRelationships: [],
    kgTemporalEvents: [],
    kgExtracting: false,
    kgJobId: null,
    kgAvailableTypes: ['person', 'organization', 'location', 'concept', 'event', 'product', 'technology'],

    // Vault browser
    vaultFiles: [],
    vaultBrowseDir: '.',

    // Admin
    tenants: [],
    tenantCollections: {},
    selectedTenant: '',
    aclTenantId: '',
    aclRole: 'reader',
    permTenantId: '',
    permCollection: '',
    permLevel: 'read',
    rateLimitTenantId: '',
    rateLimitRps: 100,
    quotaData: {},
    quotaTenantId: '',
    auditEvents: [],
    auditLoading: false,

    // Agent-GO
    agentGoUrl: localStorage.getItem('agentGoUrl') || 'http://localhost:4243',
    agentGoConnected: false,
    agentGoConnecting: false,
    agentGoHealth: null,
    agentRuns: [],
    runPollId: null,
    showAgentGo: false,
  }
}
