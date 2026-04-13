{===============================================================================
  VindexLLM™ - Graph-Walk LLM Inference Engine

  Copyright © 2026-present tinyBigGAMES™ LLC
  All Rights Reserved.

  https://vipervm.org

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
  // User Lexer Messages
  //--------------------------------------------------------------------------
  RSUserLexerUnexpectedChar      = 'Unexpected character: ''%s''';
  RSUserLexerUnterminatedString  = 'Unterminated string literal';
  RSUserLexerUnterminatedComment = 'Unterminated comment';
  RSUserLexerUnknownDirective    = 'Unknown directive: ''%s''';

  //--------------------------------------------------------------------------
  // User Parser Messages
  //--------------------------------------------------------------------------
  RSUserParserExpectedToken      = 'Expected %s but found ''%s''';
  RSUserParserNoPrefixHandler    = 'Unexpected token in expression: ''%s''';

  //--------------------------------------------------------------------------
  // Status Messages
  //--------------------------------------------------------------------------
  RSUserLexerTokenizing          = 'Tokenizing %s...';
  RSUserParserParsing            = 'Parsing %s...';
  RSUserSemanticAnalyzing        = 'Analyzing %s...';
  RSUserCodeGenEmitting          = 'Emitting %s...';
  RSEngineTargetPlatform         = 'Target: %s';
  RSEngineBuildMode              = 'Build mode: %s';
  RSEngineOptimizeLevel          = 'Optimization: %s';

implementation

end.
