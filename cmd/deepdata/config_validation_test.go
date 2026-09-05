package main

import (
	"os"
	"testing"
)

func TestEnvInt_InvalidValue_ReturnsDefault(t *testing.T) {
	os.Setenv("TEST_ENV_INT", "notanumber")
	defer os.Unsetenv("TEST_ENV_INT")

	result := envInt("TEST_ENV_INT", 42)
	if result != 42 {
		t.Errorf("expected default 42, got %d", result)
	}
}

func TestEnvInt_ValidValue(t *testing.T) {
	os.Setenv("TEST_ENV_INT", "99")
	defer os.Unsetenv("TEST_ENV_INT")

	result := envInt("TEST_ENV_INT", 42)
	if result != 99 {
		t.Errorf("expected 99, got %d", result)
	}
}
