using System;
using System.Collections;
using UnityEngine;
using Oculus.Interaction;

/// <summary>
/// Extends InteractableColorVisual to add email validation visual feedback
/// </summary>
public class EmailValidationButtonVisual : InteractableColorVisual
{
    [Serializable]
    public class ValidationColorState
    {
        public Color Color = Color.red;
        public AnimationCurve ColorCurve = AnimationCurve.EaseInOut(0, 0, 1, 1);
        public float ColorTime = 0.3f;
        public float ReturnDelay = 0.5f;
    }

    [SerializeField]
    private ValidationColorState _invalidEmailColorState = new ValidationColorState();

    private MaterialPropertyBlockEditor _materialEditor;
    private int _validationColorShaderID; // Renamed to avoid conflict with base class
    private Coroutine _validationRoutine = null;
    private Color _previousColor;
    private bool _isShowingValidationError = false;

    // Called by FetchTextureManager when email validation fails
    public void ShowInvalidEmailState()
    {
        if (_isShowingValidationError)
            return;

        // Store the current color so we can return to it
        _previousColor = GetCurrentColor();

        // Cancel any existing validation routine
        CancelValidationRoutine();

        // Start the validation visual feedback
        _validationRoutine = StartCoroutine(ShowValidationFeedback());
    }

    protected override void Awake()
    {
        base.Awake();
        _materialEditor = GetComponent<MaterialPropertyBlockEditor>();
        _validationColorShaderID = Shader.PropertyToID("_Color"); // Using renamed field
    }

    protected void OnDestroy()
    {
        // Cannot use override since the base class doesn't have a virtual OnDestroy
        CancelValidationRoutine();
    }

    private Color GetCurrentColor()
    {
        // If the MaterialPropertyBlockEditor is available, get the current color from it
        if (_materialEditor != null && _materialEditor.MaterialPropertyBlock != null)
        {
            return _materialEditor.MaterialPropertyBlock.GetColor(_validationColorShaderID);
        }

        // Default to white if we can't get the current color
        return Color.white;
    }

    private IEnumerator ShowValidationFeedback()
    {
        _isShowingValidationError = true;

        // Transition to the invalid email color
        Color startColor = _previousColor;
        Color targetColor = _invalidEmailColorState.Color;
        float timer = 0f;

        // Animate to error color
        do
        {
            timer += Time.deltaTime;
            float normalizedTimer = Mathf.Clamp01(timer / _invalidEmailColorState.ColorTime);
            float t = _invalidEmailColorState.ColorCurve.Evaluate(normalizedTimer);
            SetValidationColor(Color.Lerp(startColor, targetColor, t));

            yield return null;
        }
        while (timer <= _invalidEmailColorState.ColorTime);

        // Hold the error color for a moment
        yield return new WaitForSeconds(_invalidEmailColorState.ReturnDelay);

        // Transition back to the previous color
        timer = 0f;
        startColor = targetColor;
        targetColor = _previousColor;

        // Animate back to original color
        do
        {
            timer += Time.deltaTime;
            float normalizedTimer = Mathf.Clamp01(timer / _invalidEmailColorState.ColorTime);
            float t = _invalidEmailColorState.ColorCurve.Evaluate(normalizedTimer);
            SetValidationColor(Color.Lerp(startColor, targetColor, t));

            yield return null;
        }
        while (timer <= _invalidEmailColorState.ColorTime);

        _isShowingValidationError = false;
        _validationRoutine = null;
    }


    private void SetValidationColor(Color color)
    {
        if (_materialEditor != null && _materialEditor.MaterialPropertyBlock != null)
        {
            _materialEditor.MaterialPropertyBlock.SetColor(_validationColorShaderID, color);
            _materialEditor.UpdateMaterialPropertyBlock();
        }
    }

    private void CancelValidationRoutine()
    {
        if (_validationRoutine != null)
        {
            StopCoroutine(_validationRoutine);
            _validationRoutine = null;
        }
    }
}