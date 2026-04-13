{===============================================================================
  VindexLLM™ - Graph-Walk LLM Inference Engine

  Copyright © 2026-present tinyBigGAMES™ LLC
  All Rights Reserved.

  https://vipervm.org

  See LICENSE for license information
===============================================================================}

{$Z4}
{$A8}

{$WARN SYMBOL_DEPRECATED OFF}
{$WARN SYMBOL_PLATFORM OFF}

{$WARN UNIT_PLATFORM OFF}
{$WARN UNIT_DEPRECATED OFF}

{$INLINE AUTO}

{ Enable leak tracking for TVdxBaseObject hierarchy.
  Remove the dot to activate }
{.$DEFINE VPR_LEAK_TRACK}

{$IFNDEF WIN64}
  {$MESSAGE Error 'Unsupported platform'}
{$ENDIF}
