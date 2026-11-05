import type { SelectionResult } from "./types.js";

type EndpointLimits = Pick<SelectionResult, "contextLength" | "maxCompletionTokens">;

const endpointLimitsByModel = new Map<string, EndpointLimits>();

export function getEndpointLimits(modelId: string): EndpointLimits | undefined {
  return endpointLimitsByModel.get(modelId);
}

export function setEndpointLimits(modelId: string, result: EndpointLimits): void {
  endpointLimitsByModel.set(modelId, result);
}

export function clearEndpointLimits(): void {
  endpointLimitsByModel.clear();
}
