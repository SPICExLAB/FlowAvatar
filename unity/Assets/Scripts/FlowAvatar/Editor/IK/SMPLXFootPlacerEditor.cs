#if UNITY_EDITOR
using UnityEditor;
using UnityEngine;

[CustomEditor(typeof(SMPLXFootPlacer))]
public class SMPLXFootPlacerEditor : Editor
{
    private SMPLXFootPlacer footPlacer;
    private bool showDetectionSettings = true;
    private bool showOptimizationSettings = true;
    private bool showIKSettings = true;

    private void OnEnable()
    {
        footPlacer = (SMPLXFootPlacer)target;
    }

    public override void OnInspectorGUI()
    {
        serializedObject.Update();

        EditorGUILayout.Space(5);
        EditorGUILayout.LabelField("SMPLX Foot Placement System", EditorStyles.boldLabel);

        // Foot references
        EditorGUILayout.LabelField("Foot References", EditorStyles.boldLabel);
        EditorGUILayout.PropertyField(serializedObject.FindProperty("leftFoot"));
        EditorGUILayout.PropertyField(serializedObject.FindProperty("rightFoot"));

        EditorGUILayout.Space(10);

        // Enable/disable penetration correction
        EditorGUILayout.PropertyField(serializedObject.FindProperty("enablePenetrationCorrection"));

        EditorGUILayout.Space(5);

        // Detection settings
        showDetectionSettings = EditorGUILayout.Foldout(showDetectionSettings, "Ground Detection Settings", true);
        if (showDetectionSettings)
        {
            EditorGUI.indentLevel++;
            EditorGUILayout.PropertyField(serializedObject.FindProperty("groundLayer"));
            EditorGUILayout.PropertyField(serializedObject.FindProperty("groundHeight"));
            EditorGUILayout.PropertyField(serializedObject.FindProperty("proximityRadius"));
            EditorGUILayout.PropertyField(serializedObject.FindProperty("maxSolveIterations"));
            EditorGUI.indentLevel--;
        }

        // Optimization settings
        showOptimizationSettings = EditorGUILayout.Foldout(showOptimizationSettings, "Optimization Settings", true);
        if (showOptimizationSettings)
        {
            EditorGUI.indentLevel++;
            EditorGUILayout.PropertyField(serializedObject.FindProperty("heightThreshold"));
            EditorGUILayout.PropertyField(serializedObject.FindProperty("detectionInterval"));
            EditorGUILayout.PropertyField(serializedObject.FindProperty("showPerformanceStats"));
            EditorGUI.indentLevel--;
        }

        // IK parameters
        showIKSettings = EditorGUILayout.Foldout(showIKSettings, "IK Parameters", true);
        if (showIKSettings)
        {
            EditorGUI.indentLevel++;

            SerializedProperty ikParams = serializedObject.FindProperty("ikParameters");

            // Basic correction
            EditorGUILayout.LabelField("Correction Strength", EditorStyles.boldLabel);
            EditorGUILayout.PropertyField(ikParams.FindPropertyRelative("correctionStrength"));
            EditorGUILayout.PropertyField(ikParams.FindPropertyRelative("correctionPasses"));
            EditorGUILayout.PropertyField(ikParams.FindPropertyRelative("progressiveStrengthScale"));

            // Joint rotation scales
            EditorGUILayout.Space(5);
            EditorGUILayout.LabelField("Joint Rotation Scales", EditorStyles.boldLabel);
            EditorGUILayout.PropertyField(ikParams.FindPropertyRelative("hipRotationScale"));
            EditorGUILayout.PropertyField(ikParams.FindPropertyRelative("kneeRotationScale"));
            EditorGUILayout.PropertyField(ikParams.FindPropertyRelative("ankleRotationScale"));

            // Stability settings
            EditorGUILayout.Space(5);
            EditorGUILayout.LabelField("Stability Settings", EditorStyles.boldLabel);
            EditorGUILayout.PropertyField(ikParams.FindPropertyRelative("smoothing"));
            EditorGUILayout.PropertyField(ikParams.FindPropertyRelative("groundAlignmentStrength"));

            EditorGUI.indentLevel--;
        }

        // Debug settings
        EditorGUILayout.Space(10);
        EditorGUILayout.LabelField("Debug Visualization", EditorStyles.boldLabel);
        EditorGUILayout.PropertyField(serializedObject.FindProperty("showDebugInfo"),
            new GUIContent("Show Debug Info", "Enable/disable debug visualization"));
        EditorGUILayout.PropertyField(serializedObject.FindProperty("showEditorVisuals"),
            new GUIContent("Show Editor Visuals", "Enable/disable Scene view editor visualization"));

        // Runtime stats
        if (Application.isPlaying)
        {
            EditorGUILayout.Space(15);
            EditorGUILayout.LabelField("Runtime Statistics", EditorStyles.boldLabel);

            EditorGUILayout.LabelField("Left Ankle", EditorStyles.boldLabel);
            EditorGUI.indentLevel++;
            EditorGUILayout.LabelField("State", footPlacer.LeftAnkleState.ToString());
            EditorGUILayout.LabelField("Penetration Depth", $"{footPlacer.LeftAnklePenetrationDepth:F3}");
            EditorGUI.indentLevel--;

            EditorGUILayout.Space(2);

            EditorGUILayout.LabelField("Left Foot", EditorStyles.boldLabel);
            EditorGUI.indentLevel++;
            EditorGUILayout.LabelField("State", footPlacer.LeftFootState.ToString());
            EditorGUILayout.LabelField("Penetration Depth", $"{footPlacer.LeftFootPenetrationDepth:F3}");
            EditorGUI.indentLevel--;

            EditorGUILayout.Space(5);

            EditorGUILayout.LabelField("Right Ankle", EditorStyles.boldLabel);
            EditorGUI.indentLevel++;
            EditorGUILayout.LabelField("State", footPlacer.RightAnkleState.ToString());
            EditorGUILayout.LabelField("Penetration Depth", $"{footPlacer.RightAnklePenetrationDepth:F3}");
            EditorGUI.indentLevel--;

            EditorGUILayout.Space(2);

            EditorGUILayout.LabelField("Right Foot", EditorStyles.boldLabel);
            EditorGUI.indentLevel++;
            EditorGUILayout.LabelField("State", footPlacer.RightFootState.ToString());
            EditorGUILayout.LabelField("Penetration Depth", $"{footPlacer.RightFootPenetrationDepth:F3}");
            EditorGUI.indentLevel--;

            EditorGUILayout.Space(2);
            EditorGUILayout.LabelField("Ground Normal", $"{footPlacer.LeftGroundNormal:F2}");

            // Add test buttons
            EditorGUILayout.Space(10);
            EditorGUILayout.BeginHorizontal();
            if (GUILayout.Button("Reset System"))
            {
                footPlacer.ResetSystem();
            }
            if (GUILayout.Button("Initial Correction"))
            {
                footPlacer.InitialFootCorrection();
            }
            EditorGUILayout.EndHorizontal();
        }

        serializedObject.ApplyModifiedProperties();
    }

    public void OnSceneGUI()
    {
        if (!footPlacer.showDebugInfo || !footPlacer.showEditorVisuals || !Application.isPlaying)
            return;

        // Draw the ground plane
        Handles.color = new Color(0.5f, 0.8f, 0.5f, 0.2f);
        Vector3 center = new Vector3(footPlacer.transform.position.x, footPlacer.groundHeight, footPlacer.transform.position.z);
        float size = 2f;
        Vector3 p1 = center + new Vector3(-size, 0, -size);
        Vector3 p2 = center + new Vector3(size, 0, -size);
        Vector3 p3 = center + new Vector3(size, 0, size);
        Vector3 p4 = center + new Vector3(-size, 0, size);

        Handles.DrawSolidRectangleWithOutline(
            new Vector3[] { p1, p2, p3, p4 },
            new Color(0.5f, 0.8f, 0.5f, 0.1f),
            new Color(0.5f, 0.8f, 0.5f, 0.5f));

        // Draw detection radius for each foot
        if (footPlacer.leftFoot != null)
        {
            // For ankle
            Transform leftAnkle = footPlacer.leftFoot.parent;
            if (leftAnkle != null)
            {
                Handles.color = footPlacer.LeftAnkleState == SMPLXFootPlacer.FootState.Penetrating ? Color.red : Color.green;
                Handles.DrawWireDisc(leftAnkle.position, Vector3.up, footPlacer.proximityRadius * 0.8f);
            }

            // For foot
            Handles.color = footPlacer.LeftFootState == SMPLXFootPlacer.FootState.Penetrating ? Color.red : Color.green;
            Handles.DrawWireDisc(footPlacer.leftFoot.position, Vector3.up, footPlacer.proximityRadius);
        }

        if (footPlacer.rightFoot != null)
        {
            // For ankle
            Transform rightAnkle = footPlacer.rightFoot.parent;
            if (rightAnkle != null)
            {
                Handles.color = footPlacer.RightAnkleState == SMPLXFootPlacer.FootState.Penetrating ? Color.red : Color.green;
                Handles.DrawWireDisc(rightAnkle.position, Vector3.up, footPlacer.proximityRadius * 0.8f);
            }

            // For foot
            Handles.color = footPlacer.RightFootState == SMPLXFootPlacer.FootState.Penetrating ? Color.red : Color.green;
            Handles.DrawWireDisc(footPlacer.rightFoot.position, Vector3.up, footPlacer.proximityRadius);
        }
    }
}
#endif