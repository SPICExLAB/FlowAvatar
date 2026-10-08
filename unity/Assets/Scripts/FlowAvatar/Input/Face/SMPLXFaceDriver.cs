using System;
using System.Collections.Generic;
using UnityEngine;

using static OVRFaceExpressions;
using Newtonsoft.Json;

/// <summary>
/// Drives SMPLX face expressions using Meta's face tracking data.
/// Maps from Meta FACS expressions to ARKit blendshapes, and then from ARKit to FLAME/SMPLX expressions.
///
/// Both mapping files come from EMAGE (Liu et al., "EMAGE: Towards Unified Holistic
/// Co-Speech Gesture Generation via Expressive Masked Audio Gesture Modeling", CVPR 2024;
/// https://pantomatrix.github.io/EMAGE/, https://github.com/PantoMatrix/PantoMatrix):
/// arkit_retarget_a2e_v10.json (Meta FACS -> ARKit) and mat_final.json (EMAGE's
/// "ARKit2FLAME" weights; its FLAME jaw columns drive the jaw rotation). The ten
/// SMPL-X expression coefficients (CalculateExp000-009) are hand-tuned with
/// reference to that ARKit-to-FLAME matrix.
/// </summary>
public class SMPLXFaceDriver : MonoBehaviour
{
    [SerializeField]
    private SkinnedMeshRenderer _smplxMeshRenderer;

    [SerializeField]
    private OVRFaceExpressions _ovrFaceExpressions;

    [SerializeField]
    private TextAsset _arkitToFlameMatrixJson;

    [SerializeField]
    private TextAsset _facsToArkitMappingJson;

    [Header("Jaw Control")]
    [SerializeField]
    private Transform _jawBone;
    private Vector3 _jawClosedRotation = Vector3.zero;

    [Header("Eye Control")]
    [SerializeField]
    private Transform _leftEyeBone;

    [SerializeField]
    private Transform _rightEyeBone;

    [SerializeField]
    [Range(0f, 1f)]
    private float _eyeRotationInfluence = 1f;

    [SerializeField]
    private float _maxEyeRotation = 20f;

    [Header("Expression Settings")]
    [SerializeField]
    [Range(0f, 2f)]
    private float _expressionIntensity = 1f;

    [SerializeField]
    [Range(0f, 1f)]
    private float _expressionSmoothing = 0.2f;

    // The matrix that transforms ARKit blendshapes to FLAME expressions
    private float[,] _arkitToFlameMatrix;

    // Number of ARKit blendshapes and FLAME parameters
    private int _numArkitBlendshapes = 51;
    private int _numSmplxExpressions = 10;
    private int _numTotalFlameParameters = 103; // Including jaw params

    // ARKit blendshape values
    private float[] _arkitBlendshapeValues;

    // Mapping from Meta FACS expressions to ARKit blendshapes
    private Dictionary<FaceExpression, Dictionary<string, float>> _facsToArkitMapping;

    // List of ARKit blendshape names in the order expected by the matrix
    private string[] _arkitBlendshapeNames = new string[]
    {
        "browDownLeft", "browDownRight", "browInnerUp", "browOuterUpLeft", "browOuterUpRight",
        "cheekPuff", "cheekSquintLeft", "cheekSquintRight", "eyeBlinkLeft", "eyeBlinkRight",
        "eyeLookDownLeft", "eyeLookDownRight", "eyeLookInLeft", "eyeLookInRight",
        "eyeLookOutLeft", "eyeLookOutRight", "eyeLookUpLeft", "eyeLookUpRight",
        "eyeSquintLeft", "eyeSquintRight", "eyeWideLeft", "eyeWideRight",
        "jawForward", "jawLeft", "jawOpen", "jawRight",
        "mouthClose", "mouthDimpleLeft", "mouthDimpleRight", "mouthFrownLeft", "mouthFrownRight",
        "mouthFunnel", "mouthLeft", "mouthLowerDownLeft", "mouthLowerDownRight",
        "mouthPressLeft", "mouthPressRight", "mouthPucker", "mouthRight",
        "mouthRollLower", "mouthRollUpper", "mouthShrugLower", "mouthShrugUpper",
        "mouthSmileLeft", "mouthSmileRight", "mouthStretchLeft", "mouthStretchRight",
        "mouthUpperUpLeft", "mouthUpperUpRight", "noseSneerLeft", "noseSneerRight"
    };

    // Dictionary to map from ARKit blendshape names to their index
    private Dictionary<string, int> _arkitBlendshapeIndices;

    // Smoothed SMPLX expression values
    private float[] _currentSmplxExpressions;
    private float[] _targetSmplxExpressions;

    // Class to deserialize the ARKit to FLAME matrix
    [Serializable]
    private class MatrixData
    {
        public float[] matrix;
        public int[] shape;
    }

    void Start()
    {
        InitializeArkitToFlameMatrix();
        InitializeArkitBlendshapeIndices();
        InitializeFacsToArkitMapping();

        _arkitBlendshapeValues = new float[_numArkitBlendshapes];
        _currentSmplxExpressions = new float[_numSmplxExpressions];
        _targetSmplxExpressions = new float[_numSmplxExpressions];
    }

    void InitializeArkitToFlameMatrix()
    {
        if (_arkitToFlameMatrixJson == null)
        {
            Debug.LogError("ARKit to FLAME matrix JSON file not assigned!");
            return;
        }

        // Parse the matrix from JSON
        MatrixData matrixData = JsonUtility.FromJson<MatrixData>(_arkitToFlameMatrixJson.text);

        if (matrixData == null || matrixData.matrix == null)
        {
            Debug.LogError("Failed to parse matrix data from JSON!");
            return;
        }

        // Ensure the matrix dimensions are correct
        if (matrixData.shape.Length != 2 || matrixData.shape[0] != _numArkitBlendshapes || matrixData.shape[1] != _numTotalFlameParameters)
        {
            Debug.LogWarning($"Matrix dimensions ({matrixData.shape[0]}x{matrixData.shape[1]}) don't match expected dimensions ({_numArkitBlendshapes}x{_numTotalFlameParameters})");
        }

        // Initialize the matrix with the correct dimensions
        _arkitToFlameMatrix = new float[_numArkitBlendshapes, _numTotalFlameParameters];

        // Fill the matrix
        for (int i = 0; i < _numArkitBlendshapes; i++)
        {
            for (int j = 0; j < _numTotalFlameParameters; j++)
            {
                int index = i * _numTotalFlameParameters + j;
                if (index < matrixData.matrix.Length)
                {
                    _arkitToFlameMatrix[i, j] = matrixData.matrix[index];
                }
            }
        }

        Debug.Log("ARKit to FLAME matrix initialized successfully");
    }

    void InitializeArkitBlendshapeIndices()
    {
        _arkitBlendshapeIndices = new Dictionary<string, int>();
        for (int i = 0; i < _arkitBlendshapeNames.Length; i++)
        {
            _arkitBlendshapeIndices[_arkitBlendshapeNames[i]] = i;
        }
    }

    void InitializeFacsToArkitMapping()
    {
        _facsToArkitMapping = new Dictionary<FaceExpression, Dictionary<string, float>>();

        if (_facsToArkitMappingJson != null)
        {
            // Parse from JSON if provided
            try
            {
                _facsToArkitMapping = JsonConvert.DeserializeObject<Dictionary<FaceExpression, Dictionary<string, float>>>(_facsToArkitMappingJson.text);
                Debug.Log("Loaded FACS to ARKit mapping from JSON");
                return;
            }
            catch (Exception e)
            {
                Debug.LogError($"Error parsing FACS to ARKit mapping JSON: {e.Message}");
            }
        }

        // If no JSON or parsing failed, use hardcoded mappings from the arkit_retarget_a2e_v10.json
        AddFacsArkitMapping(FaceExpression.BrowLowererL, "browDownLeft", 1.0f);
        AddFacsArkitMapping(FaceExpression.BrowLowererR, "browDownRight", 1.0f);
        AddFacsArkitMapping(FaceExpression.CheekPuffL, "cheekPuff", 1.0f);
        AddFacsArkitMapping(FaceExpression.CheekPuffR, "cheekPuff", 1.0f);
        AddFacsArkitMapping(FaceExpression.CheekRaiserL, "cheekSquintLeft", 1.0f);
        AddFacsArkitMapping(FaceExpression.CheekRaiserR, "cheekSquintRight", 1.0f);
        AddFacsArkitMapping(FaceExpression.ChinRaiserB, "mouthShrugLower", 1.0f);
        AddFacsArkitMapping(FaceExpression.ChinRaiserT, "mouthShrugUpper", 1.0f);
        AddFacsArkitMapping(FaceExpression.DimplerL, "mouthDimpleLeft", 1.0f);
        AddFacsArkitMapping(FaceExpression.DimplerR, "mouthDimpleRight", 1.0f);
        AddFacsArkitMapping(FaceExpression.EyesClosedL, "eyeBlinkLeft", 1.0f);
        AddFacsArkitMapping(FaceExpression.EyesClosedR, "eyeBlinkRight", 1.0f);
        AddFacsArkitMapping(FaceExpression.InnerBrowRaiserL, "browInnerUp", 1.0f);
        AddFacsArkitMapping(FaceExpression.InnerBrowRaiserR, "browInnerUp", 1.0f);
        AddFacsArkitMapping(FaceExpression.OuterBrowRaiserL, "browOuterUpLeft", 1.5f);
        AddFacsArkitMapping(FaceExpression.OuterBrowRaiserR, "browOuterUpRight", 1.5f);
        AddFacsArkitMapping(FaceExpression.JawDrop, "jawOpen", 1.0f);
        AddFacsArkitMapping(FaceExpression.JawSidewaysLeft, "jawLeft", 1.0f);
        AddFacsArkitMapping(FaceExpression.JawSidewaysRight, "jawRight", 1.0f);
        AddFacsArkitMapping(FaceExpression.JawThrust, "jawForward", 1.0f);
        AddFacsArkitMapping(FaceExpression.LidTightenerL, "eyeSquintLeft", 1.0f);
        AddFacsArkitMapping(FaceExpression.LidTightenerR, "eyeSquintRight", 1.0f);
        AddFacsArkitMapping(FaceExpression.LipCornerDepressorL, "mouthFrownLeft", 1.0f);
        AddFacsArkitMapping(FaceExpression.LipCornerDepressorR, "mouthFrownRight", 1.0f);

        // Compound expression for smile
        Dictionary<string, float> smileLeft = new Dictionary<string, float>
        {
            { "cheekSquintLeft", 1.0f },
            { "eyeSquintLeft", 0.25f },
            { "mouthSmileLeft", 1.0f }
        };
        _facsToArkitMapping[FaceExpression.LipCornerPullerL] = smileLeft;

        Dictionary<string, float> smileRight = new Dictionary<string, float>
        {
            { "cheekSquintRight", 1.0f },
            { "eyeSquintRight", 0.25f },
            { "mouthSmileRight", 1.0f }
        };
        _facsToArkitMapping[FaceExpression.LipCornerPullerR] = smileRight;

        // Continue with other mappings
        AddFacsArkitMapping(FaceExpression.LipFunnelerLB, "mouthFunnel", 0.75f);
        AddFacsArkitMapping(FaceExpression.LipFunnelerLT, "mouthFunnel", 0.75f);
        AddFacsArkitMapping(FaceExpression.LipFunnelerRB, "mouthFunnel", 0.75f);
        AddFacsArkitMapping(FaceExpression.LipFunnelerRT, "mouthFunnel", 0.75f);
        AddFacsArkitMapping(FaceExpression.LipPressorL, "mouthPressLeft", 1.0f);
        AddFacsArkitMapping(FaceExpression.LipPressorR, "mouthPressRight", 1.0f);

        // Pucker compound mapping
        Dictionary<string, float> pucker = new Dictionary<string, float>
        {
            { "mouthFunnel", 0.75f },
            { "mouthPucker", 0.75f }
        };
        _facsToArkitMapping[FaceExpression.LipPuckerL] = pucker;
        _facsToArkitMapping[FaceExpression.LipPuckerR] = pucker;

        AddFacsArkitMapping(FaceExpression.LipStretcherL, "mouthStretchLeft", 1.0f);
        AddFacsArkitMapping(FaceExpression.LipStretcherR, "mouthStretchRight", 1.0f);
        AddFacsArkitMapping(FaceExpression.LipSuckLB, "mouthRollLower", 0.5f);
        AddFacsArkitMapping(FaceExpression.LipSuckLT, "mouthRollUpper", 0.5f);
        AddFacsArkitMapping(FaceExpression.LipSuckRB, "mouthRollLower", 0.5f);
        AddFacsArkitMapping(FaceExpression.LipSuckRT, "mouthRollUpper", 0.5f);
        AddFacsArkitMapping(FaceExpression.LipsToward, "mouthClose", 1.0f);
        AddFacsArkitMapping(FaceExpression.LowerLipDepressorL, "mouthLowerDownLeft", 1.0f);
        AddFacsArkitMapping(FaceExpression.LowerLipDepressorR, "mouthLowerDownRight", 1.0f);
        AddFacsArkitMapping(FaceExpression.MouthLeft, "mouthLeft", 1.0f);
        AddFacsArkitMapping(FaceExpression.MouthRight, "mouthRight", 1.0f);
        AddFacsArkitMapping(FaceExpression.NoseWrinklerL, "noseSneerLeft", 1.0f);
        AddFacsArkitMapping(FaceExpression.NoseWrinklerR, "noseSneerRight", 1.0f);
        AddFacsArkitMapping(FaceExpression.UpperLidRaiserL, "eyeWideLeft", 1.0f);
        AddFacsArkitMapping(FaceExpression.UpperLidRaiserR, "eyeWideRight", 1.0f);
        AddFacsArkitMapping(FaceExpression.UpperLipRaiserL, "mouthUpperUpLeft", 1.0f);
        AddFacsArkitMapping(FaceExpression.UpperLipRaiserR, "mouthUpperUpRight", 1.0f);

        Debug.Log("Initialized default FACS to ARKit mapping");
    }

    private void AddFacsArkitMapping(FaceExpression facsExpr, string arkitName, float weight)
    {
        if (!_facsToArkitMapping.ContainsKey(facsExpr))
        {
            _facsToArkitMapping[facsExpr] = new Dictionary<string, float>();
        }

        _facsToArkitMapping[facsExpr][arkitName] = weight;
    }

    void Update()
    {
        if (_ovrFaceExpressions == null || !_ovrFaceExpressions.ValidExpressions)
            return;

        // 1. Reset ARKit blendshape values
        for (int i = 0; i < _arkitBlendshapeValues.Length; i++)
        {
            _arkitBlendshapeValues[i] = 0f;
        }

        // 2. Map Meta FACS expressions to ARKit blendshape values
        foreach (var facsPair in _facsToArkitMapping)
        {
            FaceExpression facsExpression = facsPair.Key;
            Dictionary<string, float> arkitMappings = facsPair.Value;

            // Get the weight of this FACS expression
            float facsValue = _ovrFaceExpressions.GetWeight(facsExpression);

            if (facsValue > 0)
            {
                foreach (var arkitPair in arkitMappings)
                {
                    string arkitName = arkitPair.Key;
                    float multiplier = arkitPair.Value;

                    // Find the index for this ARKit blendshape
                    if (_arkitBlendshapeIndices.TryGetValue(arkitName, out int arkitIndex))
                    {
                        _arkitBlendshapeValues[arkitIndex] += facsValue * multiplier;
                    }
                }
            }
        }

        // 3. Apply ARKit to FLAME transformation to get SMPLX expressions - MODIFIED FOR EXP000
        // Handle Exp000 with direct mapping
        float exp000Value = CalculateExp000();
        _targetSmplxExpressions[0] = exp000Value * _expressionIntensity;

        float exp001Value = CalculateExp001();
        _targetSmplxExpressions[1] = exp001Value * _expressionIntensity;

        float exp002Value = CalculateExp002();
        _targetSmplxExpressions[2] = exp002Value * _expressionIntensity;

        float exp003Value = CalculateExp003();
        _targetSmplxExpressions[3] = exp003Value * _expressionIntensity;

        float exp004Value = CalculateExp004();
        _targetSmplxExpressions[4] = exp004Value * _expressionIntensity;

        float exp005Value = CalculateExp005();
        _targetSmplxExpressions[5] = exp005Value * _expressionIntensity;

        float exp006Value = CalculateExp006();
        _targetSmplxExpressions[6] = exp006Value * _expressionIntensity;

        float exp007Value = CalculateExp007();
        _targetSmplxExpressions[7] = exp007Value * _expressionIntensity;

        float exp008Value = CalculateExp008();
        _targetSmplxExpressions[8] = exp008Value * _expressionIntensity;

        float exp009Value = CalculateExp009();
        _targetSmplxExpressions[9] = exp009Value * _expressionIntensity;

        // 4. Apply smoothing to the expression values
        for (int i = 0; i < _numSmplxExpressions; i++)
        {
            _currentSmplxExpressions[i] = Mathf.Lerp(_currentSmplxExpressions[i], _targetSmplxExpressions[i], 1f - _expressionSmoothing);
        }

        // 5. Apply SMPLX expressions to the mesh blendshapes
        // Find Exp000-Exp009 blend shapes by name instead of assuming they're the first 10
        for (int i = 0; i < _numSmplxExpressions; i++)
        {
            string blendShapeName = $"Exp{i.ToString().PadLeft(3, '0')}";
            int blendShapeIndex = -1;

            // Find the blend shape by name
            for (int j = 0; j < _smplxMeshRenderer.sharedMesh.blendShapeCount; j++)
            {
                if (_smplxMeshRenderer.sharedMesh.GetBlendShapeName(j) == blendShapeName)
                {
                    blendShapeIndex = j;
                    break;
                }
            }

            if (blendShapeIndex != -1)
            {
                // Use the raw FLAME range value (-200 to 200)
                float value = Mathf.Clamp(_currentSmplxExpressions[i] * 200f, -200f, 200f);

                // Apply to SMPLX blendshape directly
                _smplxMeshRenderer.SetBlendShapeWeight(blendShapeIndex, value);
            }
            else
            {
                Debug.LogWarning($"Could not find blend shape {blendShapeName}");
            }
        }

        // 6. Handle jaw rotation directly
        HandleJawRotation();

        // 7. Handle eye rotation directly
        HandleEyeRotation();
    }

    //direct calculation of Exp000
    private float CalculateExp000()
    {
        // Start with a base value
        float exp000Value = 0f;

        // POSITIVE VALUES (mouth corners up/smile - values around 200)
        // Stronger influence from smile parameters
        if (_arkitBlendshapeIndices.TryGetValue("mouthSmileLeft", out int mouthSmileLeftIdx))
            exp000Value += _arkitBlendshapeValues[mouthSmileLeftIdx] * 1.5f;

        if (_arkitBlendshapeIndices.TryGetValue("mouthSmileRight", out int mouthSmileRightIdx))
            exp000Value += _arkitBlendshapeValues[mouthSmileRightIdx] * 1.5f;

        // Small contribution from dimples that accompany smiles
        if (_arkitBlendshapeIndices.TryGetValue("mouthDimpleLeft", out int mouthDimpleLeftIdx))
            exp000Value += _arkitBlendshapeValues[mouthDimpleLeftIdx] * 0.7f;

        if (_arkitBlendshapeIndices.TryGetValue("mouthDimpleRight", out int mouthDimpleRightIdx))
            exp000Value += _arkitBlendshapeValues[mouthDimpleRightIdx] * 0.7f;

        // Subtle cheek raise with smile
        if (_arkitBlendshapeIndices.TryGetValue("cheekSquintLeft", out int cheekSquintLeftIdx))
            exp000Value += _arkitBlendshapeValues[cheekSquintLeftIdx] * 0.5f;

        if (_arkitBlendshapeIndices.TryGetValue("cheekSquintRight", out int cheekSquintRightIdx))
            exp000Value += _arkitBlendshapeValues[cheekSquintRightIdx] * 0.5f;

        // NEGATIVE VALUES (mouth corners down/frown - values around -200)
        // Strong influence from frown parameters
        if (_arkitBlendshapeIndices.TryGetValue("mouthFrownLeft", out int mouthFrownLeftIdx))
            exp000Value -= _arkitBlendshapeValues[mouthFrownLeftIdx] * 1.7f;

        if (_arkitBlendshapeIndices.TryGetValue("mouthFrownRight", out int mouthFrownRightIdx))
            exp000Value -= _arkitBlendshapeValues[mouthFrownRightIdx] * 1.7f;

        // Some contribution from stretch parameters which can pull mouth corners down
        if (_arkitBlendshapeIndices.TryGetValue("mouthStretchLeft", out int mouthStretchLeftIdx))
            exp000Value -= _arkitBlendshapeValues[mouthStretchLeftIdx] * 0.5f;

        if (_arkitBlendshapeIndices.TryGetValue("mouthStretchRight", out int mouthStretchRightIdx))
            exp000Value -= _arkitBlendshapeValues[mouthStretchRightIdx] * 0.5f;

        // Additional negative influences that counter smiling
        if (_arkitBlendshapeIndices.TryGetValue("mouthPucker", out int mouthPuckerIdx))
            exp000Value -= _arkitBlendshapeValues[mouthPuckerIdx] * 0.8f;

        if (_arkitBlendshapeIndices.TryGetValue("mouthFunnel", out int mouthFunnelIdx))
            exp000Value -= _arkitBlendshapeValues[mouthFunnelIdx] * 0.6f;

        // Small bias to help reach positive values a bit more easily
        float bias = 0.1f;

        // Scale and normalize
        return Mathf.Clamp(exp000Value + bias, -1f, 1f);
    }


    // Function for direct calculation of Exp001 - represents upper mouth opening/closing
    private float CalculateExp001()
    {
        // Start with a base value
        float exp001Value = 0f;

        // Positive influences - contribute to mouth opening (increased weights)
        if (_arkitBlendshapeIndices.TryGetValue("mouthUpperUpLeft", out int mouthUpperUpLeftIdx))
            exp001Value += _arkitBlendshapeValues[mouthUpperUpLeftIdx] * 0.8f; // Increased from 0.45f

        if (_arkitBlendshapeIndices.TryGetValue("mouthUpperUpRight", out int mouthUpperUpRightIdx))
            exp001Value += _arkitBlendshapeValues[mouthUpperUpRightIdx] * 0.8f; // Increased from 0.45f

        if (_arkitBlendshapeIndices.TryGetValue("mouthShrugUpper", out int mouthShrugUpperIdx))
            exp001Value += _arkitBlendshapeValues[mouthShrugUpperIdx] * 0.6f; // Increased from 0.32f

        // Add more influence from jaw open to help reach positive values
        if (_arkitBlendshapeIndices.TryGetValue("jawOpen", out int jawOpenIdx))
            exp001Value += _arkitBlendshapeValues[jawOpenIdx] * 0.4f; // Increased from 0.15f

        // Negative influences - reduced to make positive values easier to reach
        if (_arkitBlendshapeIndices.TryGetValue("mouthClose", out int mouthCloseIdx))
            exp001Value -= _arkitBlendshapeValues[mouthCloseIdx] * 0.5f; // Reduced from 0.65f

        if (_arkitBlendshapeIndices.TryGetValue("mouthPressLeft", out int mouthPressLeftIdx))
            exp001Value -= _arkitBlendshapeValues[mouthPressLeftIdx] * 0.25f; // Reduced from 0.35f

        if (_arkitBlendshapeIndices.TryGetValue("mouthPressRight", out int mouthPressRightIdx))
            exp001Value -= _arkitBlendshapeValues[mouthPressRightIdx] * 0.25f; // Reduced from 0.35f

        if (_arkitBlendshapeIndices.TryGetValue("mouthRollUpper", out int mouthRollUpperIdx))
            exp001Value -= _arkitBlendshapeValues[mouthRollUpperIdx] * 0.25f; // Reduced from 0.35f

        // Add a small positive bias to help reach positive range
        float bias = 0.15f;

        // Scale and clamp to ensure we can reach the full range
        return Mathf.Clamp(exp001Value + bias, -1f, 1f);
    }

    private float CalculateExp002()
    {
        // Start with a base value
        float exp002Value = 0f;

        // Positive influences (surprise/eye widening - values around 200)
        if (_arkitBlendshapeIndices.TryGetValue("eyeWideLeft", out int eyeWideLeftIdx))
            exp002Value += _arkitBlendshapeValues[eyeWideLeftIdx] * 1.2f;

        if (_arkitBlendshapeIndices.TryGetValue("eyeWideRight", out int eyeWideRightIdx))
            exp002Value += _arkitBlendshapeValues[eyeWideRightIdx] * 1.2f;

        // Add some surprise elements from the eyebrows, but less than in Exp003
        if (_arkitBlendshapeIndices.TryGetValue("browInnerUp", out int browInnerUpIdx))
            exp002Value += _arkitBlendshapeValues[browInnerUpIdx] * 0.6f;


        // Negative influences (disgust/eye narrowing - values around -200)
        if (_arkitBlendshapeIndices.TryGetValue("eyeSquintLeft", out int eyeSquintLeftIdx))
            exp002Value -= _arkitBlendshapeValues[eyeSquintLeftIdx] * 0.5f;

        if (_arkitBlendshapeIndices.TryGetValue("eyeSquintRight", out int eyeSquintRightIdx))
            exp002Value -= _arkitBlendshapeValues[eyeSquintRightIdx] * 0.5f;

        if (_arkitBlendshapeIndices.TryGetValue("browDownLeft", out int browDownLeftIdx))
            exp002Value -= _arkitBlendshapeValues[browDownLeftIdx] * 0.5f; // Increased weight

        if (_arkitBlendshapeIndices.TryGetValue("browDownRight", out int browDownRightIdx))
            exp002Value -= _arkitBlendshapeValues[browDownRightIdx] * 0.5f; // Increased weight


        // Apply scaling to ensure we reach the full range
        return Mathf.Clamp(exp002Value * 1f, -1f, 1f);
    }

    private float CalculateExp003()
    {
        // Start with a base value
        float exp003Value = 0f;

        // Negative influences (eyebrows up/surprise - values around -200)
        if (_arkitBlendshapeIndices.TryGetValue("browInnerUp", out int browInnerUpIdx))
            exp003Value -= _arkitBlendshapeValues[browInnerUpIdx] * 1.3f;

        if (_arkitBlendshapeIndices.TryGetValue("browOuterUpLeft", out int browOuterUpLeftIdx))
            exp003Value -= _arkitBlendshapeValues[browOuterUpLeftIdx] * 1.1f;

        if (_arkitBlendshapeIndices.TryGetValue("browOuterUpRight", out int browOuterUpRightIdx))
            exp003Value -= _arkitBlendshapeValues[browOuterUpRightIdx] * 1.1f;

        // Positive influences (eyebrows down/anger - values around 200)
        if (_arkitBlendshapeIndices.TryGetValue("browDownLeft", out int browDownLeftIdx))
            exp003Value += _arkitBlendshapeValues[browDownLeftIdx] * 0.5f;

        if (_arkitBlendshapeIndices.TryGetValue("browDownRight", out int browDownRightIdx))
            exp003Value += _arkitBlendshapeValues[browDownRightIdx] * 0.5f;

        // Adding subtle support from eyelids to enhance the expression
        if (_arkitBlendshapeIndices.TryGetValue("eyeSquintLeft", out int eyeSquintLeftIdx))
            exp003Value += _arkitBlendshapeValues[eyeSquintLeftIdx] * 0.4f;

        if (_arkitBlendshapeIndices.TryGetValue("eyeSquintRight", out int eyeSquintRightIdx))
            exp003Value += _arkitBlendshapeValues[eyeSquintRightIdx] * 0.4f;

        // Apply scaling to ensure we reach the full range
        return Mathf.Clamp(exp003Value * 1.2f, -1f, 1f);
    }

    private float CalculateExp004()
    {
        // Start with a base value
        float exp004Value = 0f;

        // Negative influences (slightly open mouth - values around -200)
        if (_arkitBlendshapeIndices.TryGetValue("mouthLowerDownLeft", out int mouthLowerDownLeftIdx))
            exp004Value -= _arkitBlendshapeValues[mouthLowerDownLeftIdx] * 1.1f;

        if (_arkitBlendshapeIndices.TryGetValue("mouthLowerDownRight", out int mouthLowerDownRightIdx))
            exp004Value -= _arkitBlendshapeValues[mouthLowerDownRightIdx] * 1.1f;

        if (_arkitBlendshapeIndices.TryGetValue("mouthShrugLower", out int mouthShrugLowerIdx))
            exp004Value -= _arkitBlendshapeValues[mouthShrugLowerIdx] * 0.8f;

        // Minor contribution from jaw opening - less than what affects other expressions
        if (_arkitBlendshapeIndices.TryGetValue("jawOpen", out int jawOpenIdx))
            exp004Value -= _arkitBlendshapeValues[jawOpenIdx] * 0.01f;

        // Positive influences (closed/pressed mouth - values around 200)
        if (_arkitBlendshapeIndices.TryGetValue("mouthClose", out int mouthCloseIdx))
            exp004Value += _arkitBlendshapeValues[mouthCloseIdx] * 0.9f;

        if (_arkitBlendshapeIndices.TryGetValue("mouthPressLeft", out int mouthPressLeftIdx))
            exp004Value += _arkitBlendshapeValues[mouthPressLeftIdx] * 0.8f;

        if (_arkitBlendshapeIndices.TryGetValue("mouthPressRight", out int mouthPressRightIdx))
            exp004Value += _arkitBlendshapeValues[mouthPressRightIdx] * 0.8f;

        if (_arkitBlendshapeIndices.TryGetValue("mouthRollLower", out int mouthRollLowerIdx))
            exp004Value += _arkitBlendshapeValues[mouthRollLowerIdx] * 0.7f;

        // Discriminating this from Exp001 (upper mouth)
        if (_arkitBlendshapeIndices.TryGetValue("mouthUpperUpLeft", out int mouthUpperUpLeftIdx))
            exp004Value -= _arkitBlendshapeValues[mouthUpperUpLeftIdx] * 0.2f; // Small negative influence

        if (_arkitBlendshapeIndices.TryGetValue("mouthUpperUpRight", out int mouthUpperUpRightIdx))
            exp004Value -= _arkitBlendshapeValues[mouthUpperUpRightIdx] * 0.2f; // Small negative influence

        // Apply scaling to ensure we reach the full range
        return Mathf.Clamp(exp004Value * 1.2f, -1f, 1f);
    }

    private float CalculateExp005()
    {
        // Start with a base value
        float exp005Value = 0f;

        // Positive influences (kiss/pursed lips - values around 200)
        if (_arkitBlendshapeIndices.TryGetValue("mouthPucker", out int mouthPuckerIdx))
            exp005Value += _arkitBlendshapeValues[mouthPuckerIdx] * 1.2f;

        if (_arkitBlendshapeIndices.TryGetValue("mouthFrownLeft", out int mouthFrownLeftIdx))
            exp005Value += _arkitBlendshapeValues[mouthFrownLeftIdx] * 0.9f;

        if (_arkitBlendshapeIndices.TryGetValue("mouthFrownRight", out int mouthFrownRightIdx))
            exp005Value += _arkitBlendshapeValues[mouthFrownRightIdx] * 0.9f;

        // Some contribution from lip tightening
        if (_arkitBlendshapeIndices.TryGetValue("mouthStretchLeft", out int mouthStretchLeftIdx))
            exp005Value += _arkitBlendshapeValues[mouthStretchLeftIdx] * 0.5f;

        if (_arkitBlendshapeIndices.TryGetValue("mouthStretchRight", out int mouthStretchRightIdx))
            exp005Value += _arkitBlendshapeValues[mouthStretchRightIdx] * 0.5f;

        // Add subtle contribution from eyes for the squinting that often accompanies kissing
        if (_arkitBlendshapeIndices.TryGetValue("eyeSquintLeft", out int eyeSquintLeftIdx))
            exp005Value += _arkitBlendshapeValues[eyeSquintLeftIdx] * 0.4f;

        if (_arkitBlendshapeIndices.TryGetValue("eyeSquintRight", out int eyeSquintRightIdx))
            exp005Value += _arkitBlendshapeValues[eyeSquintRightIdx] * 0.4f;

        // Negative influences (relaxed lips/mouth - values around -200)
        // Subtly use mouth opening parameters as negative influences
        if (_arkitBlendshapeIndices.TryGetValue("mouthSmileLeft", out int mouthSmileLeftIdx))
            exp005Value -= _arkitBlendshapeValues[mouthSmileLeftIdx] * 0.7f;

        if (_arkitBlendshapeIndices.TryGetValue("mouthSmileRight", out int mouthSmileRightIdx))
            exp005Value -= _arkitBlendshapeValues[mouthSmileRightIdx] * 0.7f;

        if (_arkitBlendshapeIndices.TryGetValue("eyeWideLeft", out int eyeWideLeftIdx))
            exp005Value -= _arkitBlendshapeValues[eyeWideLeftIdx] * 0.6f;

        if (_arkitBlendshapeIndices.TryGetValue("eyeWideRight", out int eyeWideRightIdx))
            exp005Value -= _arkitBlendshapeValues[eyeWideRightIdx] * 0.6f;

        // A bit of jaw open contributes to negative values (opposite of kiss)
        if (_arkitBlendshapeIndices.TryGetValue("jawOpen", out int jawOpenIdx))
            exp005Value -= _arkitBlendshapeValues[jawOpenIdx] * 0.1f;

        // Apply scaling to ensure we reach the full range
        return Mathf.Clamp(exp005Value * 1.15f, -1f, 1f);
    }

    private float CalculateExp006()
    {
        // Start with a base value
        float exp006Value = 0f;

        // Get individual nose flare values from left and right
        float leftSneerValue = 0f;
        float rightSneerValue = 0f;

        if (_arkitBlendshapeIndices.TryGetValue("noseSneerLeft", out int noseSneerLeftIdx))
            leftSneerValue = _arkitBlendshapeValues[noseSneerLeftIdx];

        if (_arkitBlendshapeIndices.TryGetValue("noseSneerRight", out int noseSneerRightIdx))
            rightSneerValue = _arkitBlendshapeValues[noseSneerRightIdx];

        // Use the minimum of the two to ensure symmetry (both sides must move)
        float symmetricSneerValue = Mathf.Min(leftSneerValue, rightSneerValue);
        exp006Value += symmetricSneerValue * 1.5f;

        // Add some upper lip raising that typically accompanies nose flaring
        if (_arkitBlendshapeIndices.TryGetValue("mouthUpperUpLeft", out int mouthUpperUpLeftIdx) &&
            _arkitBlendshapeIndices.TryGetValue("mouthUpperUpRight", out int mouthUpperUpRightIdx))
        {
            float symmetricUpperLipValue = Mathf.Min(
                _arkitBlendshapeValues[mouthUpperUpLeftIdx],
                _arkitBlendshapeValues[mouthUpperUpRightIdx]);
            exp006Value += symmetricUpperLipValue * 0.6f;
        }

        // Negative influences (mouth stretching/flattening - values around -200)
        if (_arkitBlendshapeIndices.TryGetValue("mouthStretchLeft", out int mouthStretchLeftIdx) &&
            _arkitBlendshapeIndices.TryGetValue("mouthStretchRight", out int mouthStretchRightIdx))
        {
            float symmetricStretchValue = Mathf.Min(
                _arkitBlendshapeValues[mouthStretchLeftIdx],
                _arkitBlendshapeValues[mouthStretchRightIdx]);
            exp006Value -= symmetricStretchValue * 1.4f;
        }

        // Add some lip press/flattening
        if (_arkitBlendshapeIndices.TryGetValue("mouthPressLeft", out int mouthPressLeftIdx) &&
            _arkitBlendshapeIndices.TryGetValue("mouthPressRight", out int mouthPressRightIdx))
        {
            float symmetricPressValue = Mathf.Min(
                _arkitBlendshapeValues[mouthPressLeftIdx],
                _arkitBlendshapeValues[mouthPressRightIdx]);
            exp006Value -= symmetricPressValue * 0.7f;
        }

        // Apply scaling to ensure we reach the full range
        return Mathf.Clamp(exp006Value * 1.1f, -1f, 1f);
    }

    private float CalculateExp007()
    {
        // Start with a base value
        float exp007Value = 0f;

        // Positive influences (left side raise - values around 200)
        if (_arkitBlendshapeIndices.TryGetValue("mouthSmileLeft", out int mouthSmileLeftIdx))
            exp007Value += _arkitBlendshapeValues[mouthSmileLeftIdx] * 1.0f;

        if (_arkitBlendshapeIndices.TryGetValue("cheekSquintLeft", out int cheekSquintLeftIdx))
            exp007Value += _arkitBlendshapeValues[cheekSquintLeftIdx] * 0.8f;

        if (_arkitBlendshapeIndices.TryGetValue("mouthLeft", out int mouthLeftIdx))
            exp007Value += _arkitBlendshapeValues[mouthLeftIdx] * 0.7f;

        // Add some eye squint that might accompany asymmetric expression
        if (_arkitBlendshapeIndices.TryGetValue("eyeSquintLeft", out int eyeSquintLeftIdx))
            exp007Value += _arkitBlendshapeValues[eyeSquintLeftIdx] * 0.3f;

        // Negative influences (right side raise - values around -200)
        if (_arkitBlendshapeIndices.TryGetValue("mouthSmileRight", out int mouthSmileRightIdx))
            exp007Value -= _arkitBlendshapeValues[mouthSmileRightIdx] * 1.0f;

        if (_arkitBlendshapeIndices.TryGetValue("cheekSquintRight", out int cheekSquintRightIdx))
            exp007Value -= _arkitBlendshapeValues[cheekSquintRightIdx] * 0.8f;

        if (_arkitBlendshapeIndices.TryGetValue("mouthRight", out int mouthRightIdx))
            exp007Value -= _arkitBlendshapeValues[mouthRightIdx] * 0.7f;

        // Add some eye squint for right side
        if (_arkitBlendshapeIndices.TryGetValue("eyeSquintRight", out int eyeSquintRightIdx))
            exp007Value -= _arkitBlendshapeValues[eyeSquintRightIdx] * 0.3f;

        // Apply scaling to ensure we reach the full range
        return Mathf.Clamp(exp007Value * 1.2f, -1f, 1f);
    }

    private float CalculateExp008()
    {
        // Start with a base value
        float exp008Value = 0f;

        // Positive influences (left-side dominance - values around 200)
        // Mouth movement to left
        if (_arkitBlendshapeIndices.TryGetValue("mouthLeft", out int mouthLeftIdx))
            exp008Value += _arkitBlendshapeValues[mouthLeftIdx] * 1.2f;

        // Left nose flare
        if (_arkitBlendshapeIndices.TryGetValue("noseSneerLeft", out int noseSneerLeftIdx))
            exp008Value += _arkitBlendshapeValues[noseSneerLeftIdx] * 0.9f;

        // Left cheek movement
        if (_arkitBlendshapeIndices.TryGetValue("cheekSquintLeft", out int cheekSquintLeftIdx))
            exp008Value += _arkitBlendshapeValues[cheekSquintLeftIdx] * 0.7f;

        // Additional left-side mouth movements
        if (_arkitBlendshapeIndices.TryGetValue("mouthStretchLeft", out int mouthStretchLeftIdx))
            exp008Value += _arkitBlendshapeValues[mouthStretchLeftIdx] * 0.6f;

        if (_arkitBlendshapeIndices.TryGetValue("mouthDimpleLeft", out int mouthDimpleLeftIdx))
            exp008Value += _arkitBlendshapeValues[mouthDimpleLeftIdx] * 0.5f;

        // Negative influences (right-side dominance - values around -200)
        // Mouth movement to right
        if (_arkitBlendshapeIndices.TryGetValue("mouthRight", out int mouthRightIdx))
            exp008Value -= _arkitBlendshapeValues[mouthRightIdx] * 1.2f;

        // Right nose flare
        if (_arkitBlendshapeIndices.TryGetValue("noseSneerRight", out int noseSneerRightIdx))
            exp008Value -= _arkitBlendshapeValues[noseSneerRightIdx] * 0.9f;

        // Right cheek movement
        if (_arkitBlendshapeIndices.TryGetValue("cheekSquintRight", out int cheekSquintRightIdx))
            exp008Value -= _arkitBlendshapeValues[cheekSquintRightIdx] * 0.7f;

        // Additional right-side mouth movements
        if (_arkitBlendshapeIndices.TryGetValue("mouthStretchRight", out int mouthStretchRightIdx))
            exp008Value -= _arkitBlendshapeValues[mouthStretchRightIdx] * 0.6f;

        if (_arkitBlendshapeIndices.TryGetValue("mouthDimpleRight", out int mouthDimpleRightIdx))
            exp008Value -= _arkitBlendshapeValues[mouthDimpleRightIdx] * 0.5f;

        // Apply scaling to ensure we reach the full range
        return Mathf.Clamp(exp008Value * 1.1f, -1f, 1f);
    }

    private float CalculateExp009()
    {
        // Start with a base value
        float exp009Value = 0f;

        // Positive influences (closed eyes - values around 200)
        if (_arkitBlendshapeIndices.TryGetValue("eyeBlinkLeft", out int eyeBlinkLeftIdx))
            exp009Value += _arkitBlendshapeValues[eyeBlinkLeftIdx] * 0.1f;

        if (_arkitBlendshapeIndices.TryGetValue("eyeBlinkRight", out int eyeBlinkRightIdx))
            exp009Value += _arkitBlendshapeValues[eyeBlinkRightIdx] * 0.1f;

        // Additional squinting that can accompany blinking
        if (_arkitBlendshapeIndices.TryGetValue("eyeSquintLeft", out int eyeSquintLeftIdx))
            exp009Value += _arkitBlendshapeValues[eyeSquintLeftIdx] * 0.5f;

        if (_arkitBlendshapeIndices.TryGetValue("eyeSquintRight", out int eyeSquintRightIdx))
            exp009Value += _arkitBlendshapeValues[eyeSquintRightIdx] * 0.5f;

        // Negative influences (wide open eyes - values around -200)
        if (_arkitBlendshapeIndices.TryGetValue("eyeWideLeft", out int eyeWideLeftIdx))
            exp009Value -= _arkitBlendshapeValues[eyeWideLeftIdx] * 1.0f;

        if (_arkitBlendshapeIndices.TryGetValue("eyeWideRight", out int eyeWideRightIdx))
            exp009Value -= _arkitBlendshapeValues[eyeWideRightIdx] * 1.0f;


        // Apply scaling to ensure we reach the full range
        return Mathf.Clamp(exp009Value * 1.0f, -1f, 1f);
    }

    private void HandleJawRotation()
    {
        if (_jawBone == null)
            return;

        // Get jaw parameters from the ARKit-to-FLAME matrix (last 3 values of the 103)
        Vector3 jawRotation = Vector3.zero;

        for (int i = 0; i < _numArkitBlendshapes; i++)
        {
            // Get the contribution to each of the 3 jaw parameters from this ARKit blendshape
            if (_numTotalFlameParameters >= 103)
            {
                jawRotation.x += _arkitBlendshapeValues[i] * _arkitToFlameMatrix[i, 100] * 200f; // Jaw param 1
                jawRotation.y += _arkitBlendshapeValues[i] * _arkitToFlameMatrix[i, 101] * 200f; // Jaw param 2
                jawRotation.z += _arkitBlendshapeValues[i] * _arkitToFlameMatrix[i, 102] * 200f; // Jaw param 3
            }
        }

        // Scale jaw rotation to reasonable values (adjust these multipliers as needed for your model)
        Vector3 scaledRotation = new Vector3(
            jawRotation.x * 1.0f, 
            jawRotation.y * -0.8f,
            jawRotation.z * 0.6f
        );

        // Apply influence and add to base rotation
        _jawBone.localRotation = Quaternion.Euler(_jawClosedRotation + scaledRotation);
    }

    private void HandleEyeRotation()
    {
        if (_leftEyeBone == null || _rightEyeBone == null)
            return;

        // Get eye look values from FACS
        float lookLeftL = _ovrFaceExpressions.GetWeight(FaceExpression.EyesLookLeftL);
        float lookRightL = _ovrFaceExpressions.GetWeight(FaceExpression.EyesLookRightL);
        float lookUpL = _ovrFaceExpressions.GetWeight(FaceExpression.EyesLookUpL);
        float lookDownL = _ovrFaceExpressions.GetWeight(FaceExpression.EyesLookDownL);

        float lookLeftR = _ovrFaceExpressions.GetWeight(FaceExpression.EyesLookLeftR);
        float lookRightR = _ovrFaceExpressions.GetWeight(FaceExpression.EyesLookRightR);
        float lookUpR = _ovrFaceExpressions.GetWeight(FaceExpression.EyesLookUpR);
        float lookDownR = _ovrFaceExpressions.GetWeight(FaceExpression.EyesLookDownR);

        // Calculate horizontal and vertical rotations for each eye
        float leftHorizontal = (lookRightL - lookLeftL) * _maxEyeRotation * _eyeRotationInfluence;
        float leftVertical = (lookDownL - lookUpL) * _maxEyeRotation * _eyeRotationInfluence;

        //float rightHorizontal = (lookRightR - lookLeftR) * _maxEyeRotation * _eyeRotationInfluence;
        //float rightVertical = (lookDownR - lookUpR) * _maxEyeRotation * _eyeRotationInfluence;

        // Apply rotations to eye bones
        _leftEyeBone.localRotation = Quaternion.Euler(leftVertical, leftHorizontal, 0f);
        _rightEyeBone.localRotation = Quaternion.Euler(leftVertical, leftHorizontal, 0f);
    }
}