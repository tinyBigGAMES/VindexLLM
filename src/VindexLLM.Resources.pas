{===============================================================================
  VindexLLM™ - Liberating LLM inference

  Copyright © 2026-present tinyBigGAMES™ LLC
  All Rights Reserved.

  https://vindexllm.com

  See LICENSE for license information
===============================================================================}

unit VindexLLM.Resources;

{$I VindexLLM.Defines.inc}

interface

resourcestring

  //--------------------------------------------------------------------------
  // Severity Names
  //--------------------------------------------------------------------------
  RSSeverityHint    = 'Hint';
  RSSeverityWarning = 'Warning';
  RSSeverityError   = 'Error';
  RSSeverityFatal   = 'Fatal';
  RSSeverityNote    = 'Note';
  RSSeverityUnknown = 'Unknown';

  //--------------------------------------------------------------------------
  // Error Format Strings
  //--------------------------------------------------------------------------
  RSErrorFormatSimple              = '%s %s: %s';
  RSErrorFormatWithLocation        = '%s: %s %s: %s';
  RSErrorFormatRelatedSimple       = '  %s: %s';
  RSErrorFormatRelatedWithLocation = '  %s: %s: %s';

  //--------------------------------------------------------------------------
  // Fatal / I/O Messages
  //--------------------------------------------------------------------------
  RSFatalFileNotFound  = 'File not found: ''%s''';
  RSFatalFileReadError = 'Cannot read file ''%s'': %s';
  RSFatalInternalError = 'Internal error: %s';

  //--------------------------------------------------------------------------
  // Memory (TVdxMemory) Messages
  //--------------------------------------------------------------------------
  RSMemSessionNotOpen   = 'Memory session not open';
  RSMemDbPathEmpty      = 'Memory database path is empty';
  RSMemEmbedderNil      = 'Embedder is nil';
  RSMemEmbedderNotLoaded = 'Embedder is not loaded';
  RSMemEmbedderDetached = 'Attached embedder is no longer loaded — call DetachEmbeddings before unloading';
  RSMemNoEmbedder       = 'No embedder attached — call AttachEmbeddings first';
  RSMemEmbeddingMismatch = 'Embedding byte length mismatch (got %d, expected %d for dim %d)';
  RSMemWhereEmpty       = 'Empty WHERE clause — use PurgeAll instead';
  RSMemChunkInvalid     = 'AChunkTokens must be > 0';
  RSMemOverlapInvalid   = 'AOverlapTokens must be < AChunkTokens';

  //--------------------------------------------------------------------------
  // VirtualBuffer (TVdxVirtualBuffer<T>) Messages
  //--------------------------------------------------------------------------
  RSVBSizeZero          = 'Allocate size must be greater than zero';
  RSVBMappingFailed     = 'CreateFileMapping failed (Win32 error %d)';
  RSVBMapViewFailed     = 'MapViewOfFile failed (Win32 error %d)';
  RSVBAllocateException = 'Allocate raised exception: %s';
  RSVBAlignment         = 'File size %d is not aligned with element size %d';
  RSVBLoadFileException = 'LoadFromFile(''%s'') failed: %s';

  //--------------------------------------------------------------------------
  // VirtualFile (TVdxVirtualFile<T>) Messages
  //--------------------------------------------------------------------------
  RSVFNotOpen           = 'VirtualFile is not open';
  RSVFOpenFailed        = 'Cannot open file ''%s'' (Win32 error %d)';
  RSVFMappingFailed     = 'CreateFileMapping on ''%s'' failed (Win32 error %d)';
  RSVFMapViewFailed     = 'MapViewOfFile on ''%s'' failed (Win32 error %d)';
  RSVFOpenException     = 'Open(''%s'') raised exception: %s';
  RSVFEmpty             = 'File ''%s'' is empty';

  //--------------------------------------------------------------------------
  // Compute (TVdxCompute) Messages
  //--------------------------------------------------------------------------
  RSVkLibLoadFailed    = 'Failed to load vulkan-1.dll - no Vulkan driver installed';
  RSVkProcMissing      = 'Vulkan proc not found: %s';
  RSVkNoGpu            = 'No Vulkan-capable GPU found';
  RSVkNoComputeQueue   = 'No GPU with compute queue found';
  RSVkGpuIndexInvalid  = 'GPU index %d out of range (found %d device(s))';
  RSVkNoMemoryType     = 'No suitable memory type (bits=$%x, props=$%x)';
  RSVkCallFailed       = '%s failed (VkResult=%d)';

  //--------------------------------------------------------------------------
  // Shaders (VdxLoadShader) Messages
  //--------------------------------------------------------------------------
  RSShNotFound = 'Shader resource ''%s'' not found';

  //--------------------------------------------------------------------------
  // GGUFReader (TVdxGGUFReader) Messages
  //--------------------------------------------------------------------------
  RSGGReadPastEOF        = 'GGUF read past end of file: offset=%d, need=%d, filesize=%d';
  RSGGBadMagic           = 'Invalid GGUF magic: expected $%08X, got $%08X';
  RSGGUnsupportedVersion = 'Unsupported GGUF version: %d (need >= 2)';
  RSGGUnknownMetaType    = 'Unknown GGUF metadata type: %d';
  RSGGNoDataBase         = 'GGUF file not open or tensor data base not computed';
  RSGGParseException     = 'GGUF parse raised exception: %s';

  //--------------------------------------------------------------------------
  // Tokenizer (TVdxTokenizer) Messages
  //--------------------------------------------------------------------------
  RSTKNilReader      = 'Tokenizer: reader is nil';
  RSTKReaderNotOpen  = 'Tokenizer: reader is not open (call TVdxGGUFReader.Open first)';
  RSTKMissingTokens  = 'Tokenizer: metadata key ''tokenizer.ggml.tokens'' not found';
  RSTKTokensNotArray = 'Tokenizer: ''tokenizer.ggml.tokens'' has wrong type (got %d, expected array)';
  RSTKMissingScores  = 'Tokenizer: metadata key ''tokenizer.ggml.scores'' not found — BPE cannot merge without scores';
  RSTKMissingTypes   = 'Tokenizer: metadata key ''tokenizer.ggml.token_type'' not found — special tokens cannot be identified';

  //--------------------------------------------------------------------------
  // LayerNorm (TVdxLayerNorm) Messages
  //--------------------------------------------------------------------------
  RSLNNotInit          = 'LayerNorm: not initialized (call Init first)';
  RSLNAlreadyInit      = 'LayerNorm: already initialized';
  RSLNComputeNil       = 'LayerNorm.Init: ACompute is nil';
  RSLNInitException    = 'LayerNorm: Init raised exception: %s';
  RSLNTensorNotFound   = 'LayerNorm: GGUF tensor not found: %s';
  RSLNTensorWrongType  = 'LayerNorm: tensor %s expected F32 but got %s';
  RSLNUploadException  = 'LayerNorm: UploadNormWeights raised exception: %s';

implementation

end.
