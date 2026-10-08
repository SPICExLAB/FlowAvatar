using System;
using System.Collections;
using System.Collections.Generic;
using System.IO;
using System.Text;
using System.Net.Http;
using UnityEngine;
using UnityEngine.Networking;
using Newtonsoft.Json;
using Newtonsoft.Json.Linq;

/// <summary>
/// AvatarFetcher retrieves a personalised texture and beta values from a self-hosted
/// avatar-pipeline server and applies them to the SMPL-X avatar in VR. Optional: the
/// demo runs with the default avatar when no server is configured.
/// </summary>
public class AvatarFetcher : MonoBehaviour
{
    [Header("API Configuration")]
    [Tooltip("fetch-texture endpoint of your avatar-pipeline server, e.g. http://192.168.1.10:5000/api/unity/fetch-texture")]
    [SerializeField] private string apiUrl = "";

    [Header("Avatar Configuration")]
    [SerializeField] private GameObject avatarPrefab;
    [SerializeField] private string materialName = "SMPLX-Basic";
    [SerializeField] private Transform spawnPoint;

    [Header("Storage Configuration")]
    [SerializeField] private string textureSavePath = "FlowAvatarTextures";
    [SerializeField] private string betasSavePath = "FlowAvatarBetas";

    [Header("Status")]
    [SerializeField] private string currentStatus = "Ready";

    private GameObject spawnedAvatar;
    private FlowAvatarSMPLxOVR avatarModel;

    // Holds the beta values parsed from the JSON response
    private float[] betaValues = new float[10];

    private void Awake()
    {
        // Create directories if they don't exist
        EnsureDirectoryExists(textureSavePath);
        EnsureDirectoryExists(betasSavePath);
    }

    /// <summary>
    /// Initiates the avatar fetching process for the given email
    /// </summary>
    /// <param name="email">User's email address</param>
    public void FetchAvatarForUser(string email)
    {
        currentStatus = "Starting avatar fetch...";
        StartCoroutine(FetchAvatarDataCoroutine(email));
    }

    private IEnumerator FetchAvatarDataCoroutine(string email)
    {
        if (string.IsNullOrEmpty(apiUrl))
        {
            Debug.LogError("[AvatarFetcher] No avatar server configured (apiUrl is empty)");
            currentStatus = "Error: no avatar server configured";
            yield break;
        }

        currentStatus = "Connecting to avatar server...";

        // Prepare API request
        string jsonPayload = JsonUtility.ToJson(new EmailRequest { email = email });
        byte[] bodyRaw = Encoding.UTF8.GetBytes(jsonPayload);

        using (UnityWebRequest webRequest = new UnityWebRequest(apiUrl, "POST"))
        {
            webRequest.uploadHandler = new UploadHandlerRaw(bodyRaw);
            webRequest.downloadHandler = new DownloadHandlerBuffer();
            webRequest.SetRequestHeader("Content-Type", "application/json");
            webRequest.SetRequestHeader("Accept", "application/json");

            // Send the request
            yield return webRequest.SendWebRequest();

            if (webRequest.result != UnityWebRequest.Result.Success)
            {
                Debug.LogError($"Error fetching avatar data: {webRequest.error}");
                currentStatus = $"Error: {webRequest.error}";
                yield break;
            }

            // Parse the response
            string responseJson = webRequest.downloadHandler.text;
            Debug.Log($"Server response: {responseJson}");

            AvatarResponse response = null;
            try
            {
                response = JsonUtility.FromJson<AvatarResponse>(responseJson);
            }
            catch (Exception ex)
            {
                Debug.LogError($"Error parsing response: {ex.Message}");
                currentStatus = "Error parsing server response";
                yield break;
            }

            if (response == null || string.IsNullOrEmpty(response.texture_url))
            {
                Debug.LogError("Invalid response format or missing texture URL");
                currentStatus = "Invalid server response";
                yield break;
            }

            // Download texture and betas data
            yield return DownloadTextureAndBetas(response, email);
        }
    }

    private IEnumerator DownloadTextureAndBetas(AvatarResponse response, string email)
    {
        // Download texture
        currentStatus = "Downloading texture...";
        string textureFileName = $"{email.Split('@')[0]}_texture.png";
        string textureFilePath = Path.Combine(Application.persistentDataPath, textureSavePath, textureFileName);

        // Make sure directory exists
        string textureDir = Path.GetDirectoryName(textureFilePath);
        if (!Directory.Exists(textureDir))
        {
            Directory.CreateDirectory(textureDir);
        }

        using (UnityWebRequest textureRequest = UnityWebRequestTexture.GetTexture(response.texture_url))
        {
            yield return textureRequest.SendWebRequest();

            if (textureRequest.result != UnityWebRequest.Result.Success)
            {
                Debug.LogError($"Error downloading texture: {textureRequest.error}");
                currentStatus = $"Error downloading texture: {textureRequest.error}";
                yield break;
            }

            // Save texture to disk
            Texture2D texture = DownloadHandlerTexture.GetContent(textureRequest);
            byte[] textureBytes = texture.EncodeToPNG();
            File.WriteAllBytes(textureFilePath, textureBytes);
            Debug.Log($"Texture saved to: {textureFilePath}");
        }

        // Process betas data
        if (!string.IsNullOrEmpty(response.betas_url))
        {
            currentStatus = "Downloading betas data...";
            string betasFileName = $"{email.Split('@')[0]}_betas.json";
            string betasFilePath = Path.Combine(Application.persistentDataPath, betasSavePath, betasFileName);

            // Make sure directory exists
            string betasDir = Path.GetDirectoryName(betasFilePath);
            if (!Directory.Exists(betasDir))
            {
                Directory.CreateDirectory(betasDir);
            }

            // If betas_data is already in the response, use it
            if (response.betas_data != null && response.betas_data.Length > 0)
            {
                ProcessBetasData(response.betas_data, betasFilePath);
            }
            else
            {
                // Otherwise download from URL
                using (UnityWebRequest betasRequest = UnityWebRequest.Get(response.betas_url))
                {
                    yield return betasRequest.SendWebRequest();

                    if (betasRequest.result != UnityWebRequest.Result.Success)
                    {
                        Debug.LogError($"Error downloading betas data: {betasRequest.error}");
                        currentStatus = $"Error downloading betas: {betasRequest.error}";
                        // Continue with texture only
                    }
                    else
                    {
                        string betasJson = betasRequest.downloadHandler.text;
                        File.WriteAllText(betasFilePath, betasJson);
                        Debug.Log($"Betas data saved to: {betasFilePath}");

                        // Parse JSON to extract betas
                        try
                        {
                            JArray betasArray = JArray.Parse(betasJson);
                            for (int i = 0; i < Math.Min(betasArray.Count, 10); i++)
                            {
                                betaValues[i] = betasArray[i].Value<float>();
                            }
                            Debug.Log($"Parsed {betasArray.Count} beta values");
                        }
                        catch (Exception ex)
                        {
                            Debug.LogError($"Error parsing betas JSON: {ex.Message}");
                            // Continue with texture only
                        }
                    }
                }
            }
        }

        // Apply texture and betas to avatar
        ApplyTextureAndBetas(textureFilePath);
    }

    private void ProcessBetasData(float[] betasData, string savePath)
    {
        try
        {
            // Copy values to our betas array
            for (int i = 0; i < Math.Min(betasData.Length, 10); i++)
            {
                betaValues[i] = betasData[i];
            }

            // Save to file
            string betasJson = JsonConvert.SerializeObject(betasData);
            File.WriteAllText(savePath, betasJson);
            Debug.Log($"Betas data saved to: {savePath}");
        }
        catch (Exception ex)
        {
            Debug.LogError($"Error processing betas data: {ex.Message}");
        }
    }

    private void ApplyTextureAndBetas(string textureFilePath)
    {
        currentStatus = "Applying texture and body shape...";

        // Load the texture
        Texture2D texture = new Texture2D(2, 2);
        if (!File.Exists(textureFilePath))
        {
            Debug.LogError($"Texture file not found: {textureFilePath}");
            currentStatus = "Error: Texture file not found";
            return;
        }

        byte[] fileData = File.ReadAllBytes(textureFilePath);
        texture.LoadImage(fileData);
        Debug.Log($"Loaded texture: {textureFilePath}, size: {texture.width}x{texture.height}");

        // Remove previous avatar if it exists
        if (spawnedAvatar != null)
        {
            Destroy(spawnedAvatar);
        }

        // Instantiate the prefab
        if (spawnPoint != null)
        {
            spawnedAvatar = Instantiate(avatarPrefab, spawnPoint.position, spawnPoint.rotation);
        }
        else
        {
            spawnedAvatar = Instantiate(avatarPrefab, Vector3.zero, Quaternion.identity);
        }

        // Find and store the FlowAvatarSMPLxOVR component
        avatarModel = spawnedAvatar.GetComponent<FlowAvatarSMPLxOVR>();
        if (avatarModel == null)
        {
            Debug.LogError("No FlowAvatarSMPLxOVR component found on the spawned avatar");
            currentStatus = "Error: Avatar component not found";
            return;
        }

        // Apply betas if we have them
        if (betaValues != null)
        {
            Debug.Log("Applying beta values to avatar");
            for (int i = 0; i < 10; i++)
            {
                avatarModel.betas[i] = betaValues[i];
            }
            avatarModel.SetBetaShapes();

            // Snap the character to the ground after applying beta shapes
            avatarModel.SnapToGroundPlane();
            Debug.Log("Avatar snapped to ground plane after applying beta shapes");
        }

        // Apply texture
        ApplyTextureToAvatar(texture);

        currentStatus = "Avatar successfully loaded and applied";
        Debug.Log("Avatar successfully loaded and applied");
    }

    private void ApplyTextureToAvatar(Texture2D texture)
    {
        // Find the SkinnedMeshRenderer in the hierarchy
        SkinnedMeshRenderer meshRenderer = spawnedAvatar.GetComponentInChildren<SkinnedMeshRenderer>();
        if (meshRenderer == null)
        {
            Debug.LogError("No SkinnedMeshRenderer found in the avatar hierarchy");
            currentStatus = "Error: Mesh renderer not found";
            return;
        }

        // Find the specified material by name or use the first one
        Material material = null;
        int materialIndex = -1;

        // Try to find the requested material by name
        for (int i = 0; i < meshRenderer.sharedMaterials.Length; i++)
        {
            if (meshRenderer.sharedMaterials[i] != null &&
                meshRenderer.sharedMaterials[i].name.Contains(materialName))
            {
                material = meshRenderer.sharedMaterials[i];
                materialIndex = i;
                break;
            }
        }

        // If material not found, use the first one
        if (material == null && meshRenderer.sharedMaterials.Length > 0)
        {
            material = meshRenderer.sharedMaterials[0];
            materialIndex = 0;
            Debug.LogWarning($"Material {materialName} not found, using first material: {material.name}");
        }

        if (material == null)
        {
            Debug.LogError("No valid material found on the avatar");
            currentStatus = "Error: No valid material found";
            return;
        }

        // Create a new material based on the original
        Material newMaterial = new Material(material);
        newMaterial.name = $"{material.name}_WithTexture";

        // Set the texture to the main texture property
        newMaterial.mainTexture = texture;

        // Additional texture properties that might be used
        if (newMaterial.HasProperty("_BaseMap"))
            newMaterial.SetTexture("_BaseMap", texture);

        if (newMaterial.HasProperty("_MainTex"))
            newMaterial.SetTexture("_MainTex", texture);

        // Apply the new material to the renderer
        Material[] materials = meshRenderer.sharedMaterials;
        materials[materialIndex] = newMaterial;
        meshRenderer.sharedMaterials = materials;

        Debug.Log($"Applied texture to {meshRenderer.gameObject.name} with material {newMaterial.name}");
    }

    private void EnsureDirectoryExists(string path)
    {
        string fullPath = Path.Combine(Application.persistentDataPath, path);
        if (!Directory.Exists(fullPath))
        {
            Directory.CreateDirectory(fullPath);
            Debug.Log($"Created directory: {fullPath}");
        }
    }

    // Helper classes for JSON serialization
    [Serializable]
    private class EmailRequest
    {
        public string email;
    }

    [Serializable]
    public class AvatarResponse
    {
        public string texture_url;
        public string betas_url;
        public string status;
        public string user_id;
        public float[] betas_data;
    }
}